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
def _extract_score_and_justification(text):
    if text:
        text = re.sub('score', 'score', text, flags=re.IGNORECASE)
        text = re.sub('justification', 'justification', text, flags=re.IGNORECASE)
        try:
            data = json.loads(text)
            score = int(data.get('score'))
            justification = data.get('justification')
        except json.JSONDecodeError:
            if (match := re.search('score: (\\d+),?\\s*justification: (.+)', text)) or (match := re.search('\\s*score:\\s*(\\d+)\\s*justification:\\s*(.+)', text, re.DOTALL)):
                score = int(match.group(1))
                justification = match.group(2)
            else:
                score = None
                justification = f'Failed to extract score and justification. Raw output: {text}'
        if not isinstance(score, (int, float)) or not isinstance(justification, str):
            return (None, f'Failed to extract score and justification. Raw output: {text}')
        return (score, justification)
    return (None, None)