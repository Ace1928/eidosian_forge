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
@staticmethod
def _validate_text_classification_input(data):
    """
        Perform input type validation for TextClassification pipelines and casting of data
        that is manipulated internally by the MLflow model server back to a structure that
        can be used for pipeline inference.

        To illustrate the input and outputs of this function, for the following inputs to
        the pyfunc.predict() call for this pipeline type:

        "text to classify"
        ["text to classify", "other text to classify"]
        {"text": "text to classify", "text_pair": "pair text"}
        [{"text": "text", "text_pair": "pair"}, {"text": "t", "text_pair": "tp" }]

        Pyfunc processing will convert these to the following structures:

        [{0: "text to classify"}]
        [{0: "text to classify"}, {0: "other text to classify"}]
        [{"text": "text to classify", "text_pair": "pair text"}]
        [{"text": "text", "text_pair": "pair"}, {"text": "t", "text_pair": "tp" }]

        The purpose of this function is to convert them into the correct format for input
        to the pipeline (wrapping as a list has no bearing on the correctness of the
        inferred classifications):

        ["text to classify"]
        ["text to classify", "other text to classify"]
        [{"text": "text to classify", "text_pair": "pair text"}]
        [{"text": "text", "text_pair": "pair"}, {"text": "t", "text_pair": "tp" }]

        Additionally, for dict input types (the 'text' & 'text_pair' input example), the dict
        input will be JSON stringified within MLflow model serving. In order to reconvert this
        structure back into the appropriate type, we use ast.literal_eval() to convert back
        to a dict. We avoid using JSON.loads() due to pandas DataFrame conversions that invert
        single and double quotes with escape sequences that are not consistent if the string
        contains escaped quotes.
        """

    def _check_keys(payload):
        """Check if a dictionary contains only allowable keys."""
        allowable_str_keys = {'text', 'text_pair'}
        if set(payload) - allowable_str_keys and (not all((isinstance(key, int) for key in payload.keys()))):
            raise MlflowException(f'Text Classification pipelines may only define dictionary inputs with keys defined as {allowable_str_keys}')
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        _check_keys(data)
        return data
    elif isinstance(data, list):
        if all((isinstance(item, str) for item in data)):
            return data
        elif all((isinstance(item, dict) for item in data)):
            for payload in data:
                _check_keys(payload)
            if list(data[0].keys())[0] == 0:
                data = [item[0] for item in data]
            try:
                return [ast.literal_eval(s) for s in data]
            except (ValueError, SyntaxError):
                return data
        else:
            raise MlflowException('An unsupported data type has been passed for Text Classification inference. Only str, list of str, dict, and list of dict are supported.')
    else:
        raise MlflowException('An unsupported data type has been passed for Text Classification inference. Only str, list of str, dict, and list of dict are supported.')