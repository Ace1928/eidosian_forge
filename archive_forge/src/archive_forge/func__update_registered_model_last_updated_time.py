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
def _update_registered_model_last_updated_time(self, name, updated_time):
    registered_model = self.get_registered_model(name)
    registered_model.last_updated_timestamp = updated_time
    self._save_registered_model_as_meta_file(registered_model)