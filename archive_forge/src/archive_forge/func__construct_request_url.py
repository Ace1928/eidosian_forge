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
def _construct_request_url(self, task_url, default_url):
    api_type = self.request_configs.get('api_type')
    api_base = self.request_configs.get('api_base')
    if api_type in ('azure', 'azure_ad', 'azuread'):
        api_version = self.request_configs.get('api_version')
        deployment_id = self.request_configs.get('deployment_id')
        return f'{api_base}/openai/deployments/{deployment_id}/{task_url}?api-version={api_version}'
    return f'{api_base}/{task_url}' if api_base else default_url