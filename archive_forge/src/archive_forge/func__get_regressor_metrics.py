import copy
import functools
import inspect
import json
import logging
import math
import pathlib
import pickle
import shutil
import tempfile
import time
import traceback
import warnings
from collections import namedtuple
from functools import partial
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import metrics as sk_metrics
from sklearn.pipeline import Pipeline as sk_Pipeline
import mlflow
from mlflow import MlflowClient
from mlflow.entities.metric import Metric
from mlflow.exceptions import MlflowException
from mlflow.metrics import (
from mlflow.models.evaluation.artifacts import (
from mlflow.models.evaluation.base import (
from mlflow.models.utils import plot_lines
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc import _ServedPyFuncModel
from mlflow.sklearn import _SklearnModelWrapper
from mlflow.utils.file_utils import TempDir
from mlflow.utils.proto_json_utils import NumpyEncoder
from mlflow.utils.time import get_current_time_millis
def _get_regressor_metrics(y, y_pred, sample_weights):
    sum_on_target = (np.array(y) * np.array(sample_weights)).sum() if sample_weights is not None else sum(y)
    return {'example_count': len(y), 'mean_absolute_error': sk_metrics.mean_absolute_error(y, y_pred, sample_weight=sample_weights), 'mean_squared_error': sk_metrics.mean_squared_error(y, y_pred, sample_weight=sample_weights), 'root_mean_squared_error': sk_metrics.mean_squared_error(y, y_pred, sample_weight=sample_weights, squared=False), 'sum_on_target': sum_on_target, 'mean_on_target': sum_on_target / len(y), 'r2_score': sk_metrics.r2_score(y, y_pred, sample_weight=sample_weights), 'max_error': sk_metrics.max_error(y, y_pred), 'mean_absolute_percentage_error': sk_metrics.mean_absolute_percentage_error(y, y_pred, sample_weight=sample_weights)}