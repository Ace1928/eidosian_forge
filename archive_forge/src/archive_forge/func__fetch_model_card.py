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
def _fetch_model_card(model_name):
    """
    Attempts to retrieve the model card for the specified model architecture iff the
    `huggingface_hub` library is installed. If a card cannot be found in the registry or
    the library is not installed, returns None.
    """
    try:
        import huggingface_hub as hub
    except ImportError:
        _logger.warning("Unable to store ModelCard data with the saved artifact. In order to preserve this information, please install the huggingface_hub package by running 'pip install huggingingface_hub>0.10.0'")
        return
    if hasattr(hub, 'ModelCard'):
        try:
            return hub.ModelCard.load(model_name)
        except Exception as e:
            _logger.warning(f'The model card could not be retrieved from the hub due to {e}')
    else:
        _logger.warning(f"The version of huggingface_hub that is installed does not provide ModelCard functionality. You have version {hub.__version__} installed. Update huggingface_hub to >= '0.10.0' to retrieve the ModelCard data.")