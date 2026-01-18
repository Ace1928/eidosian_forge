import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
X_train = ...  # Training features
y_train = ...  # Training labels
X_test = ...  # Testing features
y_test = ...  # Testing labels

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create an experiment
experiment_name = "Random Forest Classifier"
mlflow.set_experiment(experiment_name)

# Train and evaluate multiple model versions
for n_estimators in [50, 100, 200]:
    with mlflow.start_run(run_name=f"RF_{n_estimators}"):
        # Train the model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log the model and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
