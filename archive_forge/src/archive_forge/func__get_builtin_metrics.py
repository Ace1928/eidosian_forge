import importlib
import logging
import sys
from typing import Any, Dict, List, Optional
from mlflow.exceptions import BAD_REQUEST, MlflowException
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def _get_builtin_metrics(ext_task: str) -> Dict[str, str]:
    """
    Args:
        tmpl: The template kind, e.g. `regression/v1`.

    Returns:
        The builtin metrics for the mlflow evaluation service for the model type for
        this template.
    """
    if ext_task == 'regression':
        return BUILTIN_REGRESSION_RECIPE_METRICS
    elif ext_task == 'classification/binary':
        return BUILTIN_BINARY_CLASSIFICATION_RECIPE_METRICS
    elif ext_task == 'classification/multiclass':
        return BUILTIN_MULTICLASS_CLASSIFICATION_RECIPE_METRICS
    raise MlflowException(f'No builtin metrics for template kind {ext_task}', error_code=INVALID_PARAMETER_VALUE)