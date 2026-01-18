import os
import tempfile
import time
import mlflow
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
def train_function_mlflow(config):
    setup_mlflow(config)
    width, height = (config['width'], config['height'])
    for step in range(config.get('steps', 100)):
        intermediate_score = evaluation_fn(step, width, height)
        mlflow.log_metrics(dict(mean_loss=intermediate_score), step=step)
        train.report({'iterations': step, 'mean_loss': intermediate_score})
        time.sleep(0.1)