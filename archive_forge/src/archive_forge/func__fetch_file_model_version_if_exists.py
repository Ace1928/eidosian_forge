import logging
import os
import shutil
import sys
import time
import urllib
from os.path import join
from typing import List
from mlflow.entities.model_registry import (
from mlflow.entities.model_registry.model_version_stages import (
from mlflow.environment_variables import MLFLOW_REGISTRY_DIR
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.file_utils import (
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
from mlflow.utils.validation import (
def _fetch_file_model_version_if_exists(self, name, version) -> FileModelVersion:
    _validate_model_name(name)
    _validate_model_version(version)
    registered_model_version_dir = self._get_model_version_dir(name, version)
    if not exists(registered_model_version_dir):
        raise MlflowException(f'Model Version (name={name}, version={version}) not found', RESOURCE_DOES_NOT_EXIST)
    model_version = self._get_file_model_version_from_dir(registered_model_version_dir)
    if model_version.current_stage == STAGE_DELETED_INTERNAL:
        raise MlflowException(f'Model Version (name={name}, version={version}) not found', RESOURCE_DOES_NOT_EXIST)
    return model_version