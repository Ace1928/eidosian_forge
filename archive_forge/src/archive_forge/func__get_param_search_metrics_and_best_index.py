import json
import logging
import os
import sys
import traceback
import weakref
from collections import OrderedDict, defaultdict, namedtuple
from itertools import zip_longest
from urllib.parse import urlparse
import numpy as np
import mlflow
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.spark_dataset import SparkDataset
from mlflow.entities import Metric, Param
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import (
from mlflow.utils.autologging_utils import (
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.os import is_windows
from mlflow.utils.rest_utils import (
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
def _get_param_search_metrics_and_best_index(param_search_estimator, param_search_model):
    """
    Return a tuple of `(metrics_dict, best_index)`
    `metrics_dict` is a dict of metric_name --> metric_values for each param map
    - For CrossValidatorModel, the result dict contains metrics of avg_metris and std_metrics
      for each param map.
    - For TrainValidationSplitModel, the result dict contains metrics for each param map.

    `best_index` is the best index of trials.
    """
    from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel
    metrics_dict = {}
    metric_key = param_search_estimator.getEvaluator().getMetricName()
    if isinstance(param_search_model, CrossValidatorModel):
        avg_metrics = param_search_model.avgMetrics
        metrics_dict['avg_' + metric_key] = avg_metrics
        if hasattr(param_search_model, 'stdMetrics'):
            metrics_dict['std_' + metric_key] = param_search_model.stdMetrics
    elif isinstance(param_search_model, TrainValidationSplitModel):
        avg_metrics = param_search_model.validationMetrics
        metrics_dict[metric_key] = avg_metrics
    else:
        raise RuntimeError(f'Unknown parameter search model type {type(param_search_model)}.')
    if param_search_estimator.getEvaluator().isLargerBetter():
        best_index = np.argmax(avg_metrics)
    else:
        best_index = np.argmin(avg_metrics)
    return (metrics_dict, best_index)