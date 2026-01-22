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
def _extract_output_and_other_columns(model_predictions, output_column_name):
    y_pred = None
    other_output_columns = None
    ERROR_MISSING_OUTPUT_COLUMN_NAME = 'Output column name is not specified for the multi-output model. Please set the correct output column name using the `predictions` parameter.'
    if isinstance(model_predictions, list) and all((isinstance(p, dict) for p in model_predictions)):
        if output_column_name in model_predictions[0]:
            y_pred = pd.Series([p.get(output_column_name) for p in model_predictions], name=output_column_name)
            other_output_columns = pd.DataFrame([{k: v for k, v in p.items() if k != output_column_name} for p in model_predictions])
        elif len(model_predictions[0]) == 1:
            key, value = list(model_predictions[0].items())[0]
            y_pred = pd.Series(value, name=key)
            output_column_name = key
        elif output_column_name is None:
            raise MlflowException(ERROR_MISSING_OUTPUT_COLUMN_NAME, error_code=INVALID_PARAMETER_VALUE)
        else:
            raise MlflowException(f"Output column name '{output_column_name}' is not found in the model predictions list: {model_predictions}. Please set the correct output column name using the `predictions` parameter.", error_code=INVALID_PARAMETER_VALUE)
    elif isinstance(model_predictions, pd.DataFrame):
        if output_column_name in model_predictions.columns:
            y_pred = model_predictions[output_column_name]
            other_output_columns = model_predictions.drop(columns=output_column_name)
        elif len(model_predictions.columns) == 1:
            output_column_name = model_predictions.columns[0]
            y_pred = model_predictions[output_column_name]
        elif output_column_name is None:
            raise MlflowException(ERROR_MISSING_OUTPUT_COLUMN_NAME, error_code=INVALID_PARAMETER_VALUE)
        else:
            raise MlflowException(f"Output column name '{output_column_name}' is not found in the model predictions dataframe {model_predictions.columns}. Please set the correct output column name using the `predictions` parameter.", error_code=INVALID_PARAMETER_VALUE)
    elif isinstance(model_predictions, dict):
        if output_column_name in model_predictions:
            y_pred = pd.Series(model_predictions[output_column_name], name=output_column_name)
            other_output_columns = pd.DataFrame({k: v for k, v in model_predictions.items() if k != output_column_name})
        elif len(model_predictions) == 1:
            key, value = list(model_predictions.items())[0]
            y_pred = pd.Series(value, name=key)
            output_column_name = key
        elif output_column_name is None:
            raise MlflowException(ERROR_MISSING_OUTPUT_COLUMN_NAME, error_code=INVALID_PARAMETER_VALUE)
        else:
            raise MlflowException(f"Output column name '{output_column_name}' is not found in the model predictions dict {model_predictions}. Please set the correct output column name using the `predictions` parameter.", error_code=INVALID_PARAMETER_VALUE)
    return (y_pred if y_pred is not None else model_predictions, other_output_columns, output_column_name)