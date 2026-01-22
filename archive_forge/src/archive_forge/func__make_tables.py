import copy
from datetime import datetime
from typing import Callable, Dict, Optional, Union
from packaging import version
import wandb
from wandb.sdk.lib import telemetry
def _make_tables(self):
    if self.task in ['detect', 'segment']:
        validation_columns = ['Data-Index', 'Batch-Index', 'Image', 'Mean-Confidence', 'Speed']
        train_columns = ['Epoch'] + validation_columns
        self.train_validation_table = wandb.Table(columns=['Model-Name'] + train_columns)
        self.validation_table = wandb.Table(columns=['Model-Name'] + validation_columns)
        self.prediction_table = wandb.Table(columns=['Model-Name', 'Image', 'Num-Objects', 'Mean-Confidence', 'Speed'])
    elif self.task == 'classify':
        classification_columns = ['Image', 'Predicted-Category', 'Prediction-Confidence', 'Top-5-Prediction-Categories', 'Top-5-Prediction-Confindence', 'Probabilities', 'Speed']
        validation_columns = ['Data-Index', 'Batch-Index'] + classification_columns
        validation_columns.insert(3, 'Ground-Truth-Category')
        self.train_validation_table = wandb.Table(columns=['Model-Name', 'Epoch'] + validation_columns)
        self.validation_table = wandb.Table(columns=['Model-Name'] + validation_columns)
        self.prediction_table = wandb.Table(columns=['Model-Name'] + classification_columns)
    elif self.task == 'pose':
        validation_columns = ['Data-Index', 'Batch-Index', 'Image-Ground-Truth', 'Image-Prediction', 'Num-Instances', 'Mean-Confidence', 'Speed']
        train_columns = ['Epoch'] + validation_columns
        self.train_validation_table = wandb.Table(columns=['Model-Name'] + train_columns)
        self.validation_table = wandb.Table(columns=['Model-Name'] + validation_columns)
        self.prediction_table = wandb.Table(columns=['Model-Name', 'Image-Prediction', 'Num-Instances', 'Mean-Confidence', 'Speed'])