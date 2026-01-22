import logging
import os
import pathlib
import re
import shutil
import tempfile
import urllib.parse
import zipfile
from io import BytesIO
from mlflow import tracking
from mlflow.entities import Param, SourceType
from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID, MLFLOW_RUN_ID, MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException
from mlflow.projects import _project_spec
from mlflow.tracking import fluent
from mlflow.tracking.context.default_context import _get_user
from mlflow.utils.git_utils import get_git_commit, get_git_repo_url
from mlflow.utils.mlflow_tags import (
from mlflow.utils.rest_utils import augmented_raise_for_status
def _fetch_project(uri, version=None):
    """
    Fetch a project into a local directory, returning the path to the local project directory.
    """
    parsed_uri, subdirectory = _parse_subdirectory(uri)
    use_temp_dst_dir = _is_zip_uri(parsed_uri) or not _is_local_uri(parsed_uri)
    dst_dir = tempfile.mkdtemp() if use_temp_dst_dir else _parse_file_uri(parsed_uri)
    if use_temp_dst_dir:
        _logger.info('=== Fetching project from %s into %s ===', uri, dst_dir)
    if _is_zip_uri(parsed_uri):
        parsed_uri = _parse_file_uri(parsed_uri)
        _unzip_repo(zip_file=parsed_uri if _is_local_uri(parsed_uri) else _fetch_zip_repo(parsed_uri), dst_dir=dst_dir)
    elif _is_local_uri(parsed_uri):
        if use_temp_dst_dir:
            shutil.copytree(parsed_uri, dst_dir, dirs_exist_ok=True)
        if version is not None:
            if not _is_git_repo(_parse_file_uri(parsed_uri)):
                raise ExecutionException('Setting a version is only supported for Git project URIs')
            _fetch_git_repo(parsed_uri, version, dst_dir)
    else:
        _fetch_git_repo(parsed_uri, version, dst_dir)
    res = os.path.abspath(os.path.join(dst_dir, subdirectory))
    if not os.path.exists(res):
        raise ExecutionException(f'Could not find subdirectory {subdirectory} of {dst_dir}')
    return res