import os
import re
from typing import Any, Dict
from urllib.parse import urlparse
from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.file_utils import create_tmp_dir
from mlflow.utils.rest_utils import augmented_raise_for_status, cloud_storage_http_request
def _is_path(filename: str) -> bool:
    """
    Return True if `filename` is a path, False otherwise. For example,
    "foo/bar" is a path, but "bar" is not.
    """
    return os.path.basename(filename) != filename