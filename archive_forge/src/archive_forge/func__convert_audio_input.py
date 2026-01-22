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
def _convert_audio_input(self, data):
    """
        Conversion utility for decoding the base64 encoded bytes data of a raw soundfile when
        parsed through model serving, if applicable. Direct usage of the pyfunc implementation
        outside of model serving will treat this utility as a noop.

        For reference, the expected encoding for input to Model Serving will be:

        import requests
        import base64

        response = requests.get("https://www.my.sound/a/sound/file.wav")
        encoded_audio = base64.b64encode(response.content).decode("ascii")

        inference_data = json.dumps({"inputs": [encoded_audio]})

        or

        inference_df = pd.DataFrame(
        pd.Series([encoded_audio], name="audio_file")
        )
        split_dict = {"dataframe_split": inference_df.to_dict(orient="split")}
        split_json = json.dumps(split_dict)

        or

        records_dict = {"dataframe_records": inference_df.to_dict(orient="records")}
        records_json = json.dumps(records_dict)

        This utility will convert this JSON encoded, base64 encoded text back into bytes for
        input into the AutomaticSpeechRecognitionPipeline for inference.
        """

    def is_base64(s):
        try:
            return base64.b64encode(base64.b64decode(s)) == s
        except binascii.Error:
            return False

    def decode_audio(encoded):
        if isinstance(encoded, str):
            return encoded
        elif isinstance(encoded, bytes):
            if not is_base64(encoded):
                return encoded
            else:
                return base64.b64decode(encoded)
        else:
            try:
                return base64.b64decode(encoded)
            except binascii.Error as e:
                raise MlflowException("The encoded soundfile that was passed has not been properly base64 encoded. Please ensure that the raw sound bytes have been processed with `base64.b64encode(<audio data bytes>).decode('ascii')`") from e
    if isinstance(data, list) and all((isinstance(element, dict) for element in data)):
        encoded_audio = list(data[0].values())[0]
        if isinstance(encoded_audio, str):
            self._validate_str_input_uri_or_file(encoded_audio)
        return decode_audio(encoded_audio)
    elif isinstance(data, str):
        self._validate_str_input_uri_or_file(data)
    elif isinstance(data, bytes):
        return decode_audio(data)
    return data