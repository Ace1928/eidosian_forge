from __future__ import annotations
import ast
import base64
import binascii
import contextlib
import copy
import functools
import importlib
import json
import logging
import os
import pathlib
import re
import shutil
import string
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
from mlflow import pyfunc
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _get_root_uri_and_artifact_path
from mlflow.transformers.flavor_config import (
from mlflow.transformers.hub_utils import is_valid_hf_repo_id
from mlflow.transformers.llm_inference_utils import (
from mlflow.transformers.model_io import (
from mlflow.transformers.peft import (
from mlflow.transformers.signature import (
from mlflow.transformers.torch_utils import _TORCH_DTYPE_KEY, _deserialize_torch_dtype
from mlflow.types.utils import _validate_input_dictionary_contains_only_strings_and_lists_of_strings
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import (
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.logging_utils import suppress_logs
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _build_pipeline_from_model_input(model_dict: Dict[str, Any], task: Optional[str]) -> Pipeline:
    """
    Utility for generating a pipeline from component parts. If required components are not
    specified, use the transformers library pipeline component validation to force raising an
    exception. The underlying Exception thrown in transformers is verbose enough for diagnosis.
    """
    from transformers import FlaxPreTrainedModel, PreTrainedModel, TFPreTrainedModel, pipeline
    model = model_dict[FlavorKey.MODEL]
    if not (isinstance(model, (TFPreTrainedModel, PreTrainedModel, FlaxPreTrainedModel)) or is_peft_model(model)):
        raise MlflowException('The supplied model type is unsupported. The model must be one of: PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel, or PeftModel', error_code=INVALID_PARAMETER_VALUE)
    if task is None or task.startswith(_LLM_INFERENCE_TASK_PREFIX):
        default_task = _get_default_task_for_llm_inference_task(task)
        task = _get_task_for_model(model.name_or_path, default_task=default_task)
    try:
        with suppress_logs('transformers.pipelines.base', filter_regex=_PEFT_PIPELINE_ERROR_MSG):
            return pipeline(task=task, **model_dict)
    except Exception as e:
        raise MlflowException('The provided model configuration cannot be created as a Pipeline. Please verify that all required and compatible components are specified with the correct keys.', error_code=INVALID_PARAMETER_VALUE) from e