import logging
import os
import posixpath
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import docker
from mlflow import tracking
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import ExecutionException
from mlflow.projects.utils import MLFLOW_DOCKER_WORKDIR_PATH
from mlflow.utils import file_utils, process
from mlflow.utils.databricks_utils import get_databricks_env_vars
from mlflow.utils.file_utils import _handle_readonly_on_windows
from mlflow.utils.git_utils import get_git_commit
from mlflow.utils.mlflow_tags import MLFLOW_DOCKER_IMAGE_ID, MLFLOW_DOCKER_IMAGE_URI
def _get_local_uri_or_none(uri):
    if uri == 'databricks':
        return (None, None)
    parsed_uri = urllib.parse.urlparse(uri)
    if not parsed_uri.netloc and parsed_uri.scheme in ('', 'file', 'sqlite'):
        path = urllib.request.url2pathname(parsed_uri.path)
        if parsed_uri.scheme == 'sqlite':
            uri = file_utils.path_to_local_sqlite_uri(_MLFLOW_DOCKER_TRACKING_DIR_PATH)
        else:
            uri = file_utils.path_to_local_file_uri(_MLFLOW_DOCKER_TRACKING_DIR_PATH)
        return (path, uri)
    else:
        return (None, None)