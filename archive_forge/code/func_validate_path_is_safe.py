import os
import pathlib
import posixpath
import re
import urllib.parse
import uuid
from typing import Any, Tuple
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.os import is_windows
from mlflow.utils.validation import _validate_db_type_string
def validate_path_is_safe(path):
    """
    Validates that the specified path is safe to join with a trusted prefix. This is a security
    measure to prevent path traversal attacks.
    A valid path should:
        not contain separators other than '/'
        not contain .. to navigate to parent dir in path
        not be an absolute path
    """
    from mlflow.utils.file_utils import local_file_uri_to_path
    path = _decode(path)
    exc = MlflowException('Invalid path', error_code=INVALID_PARAMETER_VALUE)
    if '#' in path:
        raise exc
    if is_file_uri(path):
        path = local_file_uri_to_path(path)
    if any((s in path for s in _OS_ALT_SEPS)) or '..' in path.split('/') or pathlib.PureWindowsPath(path).is_absolute() or pathlib.PurePosixPath(path).is_absolute() or (is_windows() and len(path) >= 2 and (path[1] == ':')):
        raise exc
    return path