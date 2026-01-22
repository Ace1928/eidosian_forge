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
def _get_resource_files(self, root_dir, subfolder_name):
    source_dirs = find(root_dir, subfolder_name, full_path=True)
    if len(source_dirs) == 0:
        return (root_dir, [])
    file_names = []
    for root, _, files in os.walk(source_dirs[0]):
        for name in files:
            abspath = join(root, name)
            file_names.append(os.path.relpath(abspath, source_dirs[0]))
    if sys.platform == 'win32':
        from mlflow.utils.file_utils import relative_path_to_artifact_path
        file_names = [relative_path_to_artifact_path(x) for x in file_names]
    return (source_dirs[0], file_names)