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
def _evaluate_metric(metric_tuple, eval_fn_args):
    """
    This function calls the metric function and performs validations on the returned
    result to ensure that they are in the expected format. It will warn and will not log metrics
    that are in the wrong format.

    Args:
        extra_metric_tuple: Containing a user provided function and its index in the
            ``extra_metrics`` parameter of ``mlflow.evaluate``
        eval_fn_args: A dictionary of args needed to compute the eval metrics.

    Returns:
        MetricValue
    """
    if metric_tuple.index < 0:
        exception_header = f"Did not log builtin metric '{metric_tuple.name}' because it"
    else:
        exception_header = f"Did not log metric '{metric_tuple.name}' at index {metric_tuple.index} in the `extra_metrics` parameter because it"
    metric = metric_tuple.function(*eval_fn_args)
    if metric is None:
        _logger.warning(f'{exception_header} returned None.')
        return
    if _is_numeric(metric):
        return MetricValue(aggregate_results={metric_tuple.name: metric})
    if not isinstance(metric, MetricValue):
        _logger.warning(f'{exception_header} did not return a MetricValue.')
        return
    scores = metric.scores
    justifications = metric.justifications
    aggregates = metric.aggregate_results
    if scores is not None:
        if not isinstance(scores, list):
            _logger.warning(f'{exception_header} must return MetricValue with scores as a list.')
            return
        if any((not (_is_numeric(score) or _is_string(score) or score is None) for score in scores)):
            _logger.warning(f'{exception_header} must return MetricValue with numeric or string scores.')
            return
    if justifications is not None:
        if not isinstance(justifications, list):
            _logger.warning(f'{exception_header} must return MetricValue with justifications as a list.')
            return
        if any((not (_is_string(jus) or jus is None) for jus in justifications)):
            _logger.warning(f'{exception_header} must return MetricValue with string justifications.')
            return
    if aggregates is not None:
        if not isinstance(aggregates, dict):
            _logger.warning(f'{exception_header} must return MetricValue with aggregate_results as a dict.')
            return
        if any((not (isinstance(k, str) and (_is_numeric(v) or v is None)) for k, v in aggregates.items())):
            _logger.warning(f'{exception_header} must return MetricValue with aggregate_results with str keys and numeric values.')
            return
    return metric