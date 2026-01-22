import itertools
import logging
import os
import warnings
from string import Formatter
from typing import Any, Dict, Optional, Set
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_OPENAI_SECRET_SCOPE
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import ColSpec, Schema, TensorSpec
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.openai_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _get_task_name(task):
    mapping = _get_obj_to_task_mapping()
    if isinstance(task, str):
        if task not in mapping.values():
            raise mlflow.MlflowException(f'Unsupported task: {task}', error_code=INVALID_PARAMETER_VALUE)
        return task
    else:
        task_name = mapping.get(task)
        if task_name is None:
            raise mlflow.MlflowException(f'Unsupported task object: {task}', error_code=INVALID_PARAMETER_VALUE)
        return task_name