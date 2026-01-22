import contextlib
import functools
import importlib.util
import logging
import os
import sys
import uuid
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union
import cloudpickle
import pandas as pd
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.exceptions import MlflowException
from mlflow.langchain._langchain_autolog import (
from mlflow.langchain._rag_utils import _CODE_CONFIG, _CODE_PATH, _set_config_path
from mlflow.langchain.databricks_dependencies import (
from mlflow.langchain.runnables import _load_runnables, _save_runnables
from mlflow.langchain.utils import (
from mlflow.models import Model, ModelInputExample, ModelSignature, get_model_info
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow.utils.annotations import deprecated, experimental
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
@contextmanager
def _config_path_context(code_path: Optional[str]=None):
    _set_config_path(code_path)
    try:
        yield
    finally:
        _set_config_path(None)