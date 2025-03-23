import os
import tensorflow as tf
import pandas as pd
import numpy as np
import mlflow
import h2o
import optuna
from h2o.automl import H2OAutoML
from h2o.estimators import (
    H2ORandomForestEstimator, H2OGradientBoostingEstimator,
    H2OGeneralizedLinearEstimator, H2ONaiveBayesEstimator
)
from ydata_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ✅ Initialize H2O
h2o.init(port=55555)

# ✅ Load Fashion MNIST Dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


# Convert dataset to a pandas DataFrame
train_df = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))  # Flatten images
test_df = pd.DataFrame(X_test.reshape(X_test.shape[0], -1))

# Add target labels
train_df['label'] = y_train
test_df['label'] = y_test

# Combine train and test data
full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# Reduce the number of features to 25 randomly selected features
selected_features = np.random.choice(full_df.columns[:-1], size=25, replace=False)
full_df_reduced = full_df[list(selected_features) + ['label']]

# Generate the report with fewer features
report = ProfileReport(full_df_reduced, explorative=True)

# Save the report to an HTML file
report_path = "fashion_mnist_eda_report.html"
report.to_file(report_path)

if os.path.exists(report_path):
    print(f"✅ EDA report exists at: {os.path.abspath(report_path)}")
else:
    print("❌ EDA report NOT found! Check the file path.")

# Log EDA Report to MLflow
mlflow.set_experiment("fashion_mnist_mlops_pipeline_assignement2")
with mlflow.start_run(run_name="EDA"):
    mlflow.log_artifact(os.path.abspath(report_path))
    print("✅ EDA Report Logged to MLflow")
    mlflow.end_run()

# ===============================
# Feature scaling
# ===============================
# ✅ Scale  & Limit Data (First 250 for Training, First 10 for Testing)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))[:250]
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1))[:10]

y_train, y_test = y_train[:250], y_test[:10]

# ===============================
# Train a Random Forest classifier
# ===============================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Test Accuracy: {accuracy:.2f}")

# ===============================
# Generate SHAP Visualizations
# ===============================
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_scaled)

# Define feature names for clarity
feature_names = [f"pixel_{i}" for i in range(X_train_scaled.shape[1])]

# Create a directory to store artifact images if it doesn't exist
artifact_dir = "mlflow_artifacts"
os.makedirs(artifact_dir, exist_ok=True)

# --- SHAP Summary Plot ---
plt.figure()
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
summary_plot_path = os.path.join(artifact_dir, "shap_summary.png")
plt.savefig(summary_plot_path, bbox_inches="tight")
plt.close()
print("SHAP plots saved as artifacts.")


# Log SHAP artifacts
with mlflow.start_run(run_name="Feature Engineering with SHAP"):
    mlflow.log_param("Model", "RandomForest")
    mlflow.log_metric("Test Accuracy", accuracy)
    
    # Log SHAP plots as artifacts
    mlflow.log_artifact(summary_plot_path, artifact_path="shap_plots")

    print("SHAP artifacts logged to MLflow.")

# ===============================
# AUTO ML using H2o.ai
# ===============================
# ✅ Create DataFrame with Labels
train_df = pd.DataFrame(X_train_scaled)
train_df["label"] = y_train

test_df = pd.DataFrame(X_test_scaled)
test_df["label"] = y_test

# ✅ Convert DataFrame to H2O Frame
train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

# ✅ Define Predictors & Target
target = "label"
predictors = train_h2o.columns.remove(target)

# ✅ Convert Target to Categorical
train_h2o[target] = train_h2o[target].asfactor()
test_h2o[target] = test_h2o[target].asfactor()

# ✅ Run H2O AutoML
with mlflow.start_run(run_name="H2O AutoML Run"):
    aml = H2OAutoML(max_models=10, seed=42, max_runtime_secs=100, exclude_algos=["StackedEnsemble", "DeepLearning"])
    aml.train(x=predictors, y=target, training_frame=train_h2o)

    best_model = aml.leader  # Get best model
    best_model_type = best_model.algo  # Get best model algorithm name

    # ✅ Log Best Model Details
    mlflow.log_param("Best Model", best_model.model_id)
    mlflow.log_metric("Best Model Accuracy", best_model.auc())

print(f"🏆 Best Model from H2O AutoML: {best_model.model_id} ({best_model_type})")

# ✅ Predict & Evaluate Best Model
h2o_predictions_df = best_model.predict(test_h2o).as_data_frame()
if "predict" in h2o_predictions_df.columns:
    preds = h2o_predictions_df["predict"].values
else:
    # Option 2: Use the first column if header is not available
    preds = h2o_predictions_df.iloc[:, 0].values
h2o_accuracy = accuracy_score(y_test, preds)
print(f"✅ H2O AutoML Accuracy: {h2o_accuracy}")

# ✅ Define Optuna Objective Functions for Different Models
def objective_rf(trial):
    """Optimize H2O Random Forest."""
    model = H2ORandomForestEstimator(
        ntrees=trial.suggest_int("ntrees", 10, 100),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        min_rows=trial.suggest_int("min_rows", 1, 10),
        seed=42
    )
    model.train(x=predictors, y=target, training_frame=train_h2o)
    preds_df = model.predict(test_h2o).as_data_frame()
    if "predict" in preds_df.columns:
        preds = preds_df["predict"].values
    else:
        # Option 2: Use the first column if header is not available
        preds = preds_df.iloc[:, 0].values
    return accuracy_score(y_test, preds)

def objective_gbm(trial):
    """Optimize H2O Gradient Boosting."""
    model = H2OGradientBoostingEstimator(
        ntrees=trial.suggest_int("ntrees", 10, 100),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        learn_rate=trial.suggest_float("learn_rate", 0.01, 0.3),
        sample_rate=trial.suggest_float("sample_rate", 0.5, 1.0),
        seed=42
    )
    model.train(x=predictors, y=target, training_frame=train_h2o)
    preds_df = model.predict(test_h2o).as_data_frame()
    if "predict" in preds_df.columns:
        preds = preds_df["predict"].values
    else:
        # Option 2: Use the first column if header is not available
        preds = preds_df.iloc[:, 0].values
    return accuracy_score(y_test, preds)

def objective_glm(trial):
    """Optimize H2O Generalized Linear Model."""
    model = H2OGeneralizedLinearEstimator(
        alpha=trial.suggest_float("alpha", 0.0, 1.0),
        lambda_=trial.suggest_loguniform("lambda", 1e-5, 1e-1),
        family="multinomial"
    )
    model.train(x=predictors, y=target, training_frame=train_h2o)
    preds_df = model.predict(test_h2o).as_data_frame()
    if "predict" in preds_df.columns:
        preds = preds_df["predict"].values
    else:
        # Option 2: Use the first column if header is not available
        preds = preds_df.iloc[:, 0].values
    return accuracy_score(y_test, preds)

def objective_nb(trial):
    """Optimize H2O Naive Bayes."""
    model = H2ONaiveBayesEstimator(
        laplace=trial.suggest_float("laplace", 0.0, 1.0),
        min_sdev=trial.suggest_float("min_sdev", 1e-4, 0.1)
    )
    model.train(x=predictors, y=target, training_frame=train_h2o)
    preds_df = model.predict(test_h2o).as_data_frame()
    if "predict" in preds_df.columns:
        preds = preds_df["predict"].values
    else:
        # Option 2: Use the first column if header is not available
        preds = preds_df.iloc[:, 0].values
    return accuracy_score(y_test, preds)

# ✅ Select Correct Objective Function Based on AutoML Best Model
model_objectives = {
    "drf": objective_rf,
    "gbm": objective_gbm,
    "glm": objective_glm,
    "naivebayes": objective_nb
}

if best_model_type in model_objectives:
    print(f"🚀 Running Optuna Hyperparameter Tuning for Best Model: {best_model_type.upper()}")

    # ✅ Run Optuna for Best Model
    study = optuna.create_study(direction="maximize")
    study.optimize(model_objectives[best_model_type], n_trials=5)

    # ✅ Log Best Parameters to MLflow
    with mlflow.start_run(run_name=f"Optuna Best Model - {best_model_type.upper()}"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("Optimized Accuracy", study.best_value)

    print(f"🎯 Best Hyperparameters for {best_model_type.upper()}: {study.best_params}")
else:
    print(f"⚠️ Optuna tuning not available for model type: {best_model_type.upper()}")

# ===============================
# Detect drift using Evidently
# ===============================
# Create a drift report comparing reference (train_df) to current (test_df) data.
train_df.columns = train_df.columns.astype(str)
test_df.columns = test_df.columns.astype(str)

drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=train_df.drop("label", axis=1),
                 current_data=test_df.drop("label", axis=1))

# Save the drift report as an HTML file
drift_report_path = "drift_report.html"
drift_report.save_html(drift_report_path)
print("Data drift report generated and saved.")

# Log Performance & Drift Report to MLflow
with mlflow.start_run(run_name="Drift Detection"):
    mlflow.log_artifact(drift_report_path, artifact_path="drift_reports")    
    print("Performance metrics and drift report logged to MLflow.")