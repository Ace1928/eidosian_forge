import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from mlflow.exceptions import MlflowException
from mlflow.metrics.base import MetricValue
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.utils import _get_default_model, _get_latest_metric_version
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import (
from mlflow.utils.annotations import experimental
from mlflow.utils.class_utils import _get_class_from_string
def aggregate_function(aggregate_option, scores):
    import numpy as np
    options = {'min': np.min, 'max': np.max, 'mean': np.mean, 'median': np.median, 'variance': np.var, 'p90': lambda x: np.percentile(x, 90) if x else None}
    if aggregate_option not in options:
        raise MlflowException(message=f'Invalid aggregate option {aggregate_option}.', error_code=INVALID_PARAMETER_VALUE)
    return options[aggregate_option](scores)