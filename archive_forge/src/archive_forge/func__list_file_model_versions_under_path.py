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
def _list_file_model_versions_under_path(self, path) -> List[FileModelVersion]:
    model_versions = []
    model_version_dirs = list_all(path, filter_func=lambda x: os.path.isdir(x) and os.path.basename(os.path.normpath(x)).startswith('version-'), full_path=True)
    for directory in model_version_dirs:
        model_versions.append(self._get_file_model_version_from_dir(directory))
    return model_versions