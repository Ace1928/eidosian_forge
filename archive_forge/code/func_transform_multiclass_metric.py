import importlib
import logging
import sys
from typing import Any, Dict, List, Optional
from mlflow.exceptions import BAD_REQUEST, MlflowException
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def transform_multiclass_metric(metric_name: str, ext_task: str) -> str:
    if ext_task == 'classification/multiclass':
        for m in BUILTIN_MULTICLASS_CLASSIFICATION_RECIPE_METRICS:
            if metric_name in m.name:
                return m.name
    return metric_name