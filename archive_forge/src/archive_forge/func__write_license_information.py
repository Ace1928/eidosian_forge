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
def _write_license_information(model_name, card_data, path):
    """Writes the license file or instructions to retrieve license information."""
    fallback = f"A license file could not be found for the '{model_name}' repository. \nTo ensure that you are in compliance with the license requirements for this model, please visit the model repository here: https://huggingface.co/{model_name}"
    if (license_file := _extract_license_file_from_repository(model_name)):
        try:
            import huggingface_hub as hub
            license_location = hub.hf_hub_download(repo_id=model_name, filename=license_file)
        except Exception as e:
            _logger.warning(f'Failed to download the license file due to: {e}')
        else:
            local_license_path = pathlib.Path(license_location)
            target_path = path.joinpath(local_license_path.name)
            try:
                shutil.copy(local_license_path, target_path)
                return
            except Exception as e:
                _logger.warning(f'The license file could not be copied due to: {e}')
    if card_data and card_data.data.license != 'other':
        fallback = f"{fallback}\nThe declared license type is: '{card_data.data.license}'"
    else:
        _logger.warning('Unable to find license information for this model. Please verify permissible usage for the model you are storing prior to use.')
    path.joinpath(_LICENSE_FILE_NAME).write_text(fallback, encoding='utf-8')