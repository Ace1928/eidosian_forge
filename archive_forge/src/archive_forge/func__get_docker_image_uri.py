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
def _get_docker_image_uri(repository_uri, work_dir):
    """
    Args:
        repository_uri: The URI of the Docker repository with which to tag the image. The
            repository URI is used as the prefix of the image URI.
        work_dir: Path to the working directory in which to search for a git commit hash
    """
    repository_uri = repository_uri if repository_uri else 'docker-project'
    git_commit = get_git_commit(work_dir)
    version_string = ':' + git_commit[:7] if git_commit else ''
    return repository_uri + version_string