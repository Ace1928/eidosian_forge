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
def _convert_image_input(self, input_data):
    """
        Conversion utility for decoding the base64 encoded bytes data of a raw image file when
        parsed through model serving, if applicable. Direct usage of the pyfunc implementation
        outside of model serving will treat this utility as a noop.

        For reference, the expected encoding for input to Model Serving will be:

        import requests
        import base64

        response = requests.get("https://www.my.images/a/sound/file.jpg")
        encoded_image = base64.b64encode(response.content).decode("utf-8")

        inference_data = json.dumps({"inputs": [encoded_image]})

        or

        inference_df = pd.DataFrame(
        pd.Series([encoded_image], name="image_file")
        )
        split_dict = {"dataframe_split": inference_df.to_dict(orient="split")}
        split_json = json.dumps(split_dict)

        or

        records_dict = {"dataframe_records": inference_df.to_dict(orient="records")}
        records_json = json.dumps(records_dict)

        This utility will convert this JSON encoded, base64 encoded text back into bytes for
        input into the Image pipelines for inference.
        """

    def process_input_element(input_element):
        input_value = next(iter(input_element.values()))
        if isinstance(input_value, str) and (not self.is_base64_image(input_value)):
            self._validate_str_input_uri_or_file(input_value)
        return input_value
    if isinstance(input_data, list) and all((isinstance(element, dict) for element in input_data)):
        return [process_input_element(element) for element in input_data]
    elif isinstance(input_data, str) and (not self.is_base64_image(input_data)):
        self._validate_str_input_uri_or_file(input_data)
    return input_data