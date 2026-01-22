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
def _flatten_zero_shot_text_classifier_output_to_df(self, data):
    """
        Converts the output of sequences, labels, and scores to a Pandas DataFrame output.

        Example input:

        [{'sequence': 'My dog loves to eat spaghetti',
          'labels': ['happy', 'sad'],
          'scores': [0.9896970987319946, 0.010302911512553692]},
         {'sequence': 'My dog hates going to the vet',
          'labels': ['sad', 'happy'],
          'scores': [0.957074761390686, 0.042925238609313965]}]

        Output:

        pd.DataFrame in a fully normalized (flattened) format with each sequence, label, and score
        having a row entry.
        For example, here is the DataFrame output:

                                sequence labels    scores
        0  My dog loves to eat spaghetti  happy  0.989697
        1  My dog loves to eat spaghetti    sad  0.010303
        2  My dog hates going to the vet    sad  0.957075
        3  My dog hates going to the vet  happy  0.042925
        """
    if isinstance(data, list) and (not all((isinstance(item, dict) for item in data))):
        raise MlflowException(f'Encountered an unknown return type from the pipeline type {type(self.pipeline).__name__}. Expecting a List[Dict]', error_code=BAD_REQUEST)
    if isinstance(data, dict):
        data = [data]
    flattened_data = []
    for entry in data:
        for label, score in zip(entry['labels'], entry['scores']):
            flattened_data.append({'sequence': entry['sequence'], 'labels': label, 'scores': score})
    return pd.DataFrame(flattened_data)