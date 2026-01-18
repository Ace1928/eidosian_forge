import os
import tempfile
import time
import mlflow
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
def tune_with_setup(mlflow_tracking_uri, finish_fast=False):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name='mixin_example')
    tuner = tune.Tuner(train_function_mlflow, run_config=train.RunConfig(name='mlflow'), tune_config=tune.TuneConfig(num_samples=5), param_space={'width': tune.randint(10, 100), 'height': tune.randint(0, 100), 'steps': 5 if finish_fast else 100, 'mlflow': {'experiment_name': 'mixin_example', 'tracking_uri': mlflow.get_tracking_uri()}})
    tuner.fit()