import importlib
import logging
import sys
from typing import Any, Dict, List, Optional
from mlflow.exceptions import BAD_REQUEST, MlflowException
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def transform_multiclass_metrics_dict(eval_metrics: Dict[str, Any], ext_task) -> Dict[str, Any]:
    return {transform_multiclass_metric(k, ext_task): v for k, v in eval_metrics.items()}