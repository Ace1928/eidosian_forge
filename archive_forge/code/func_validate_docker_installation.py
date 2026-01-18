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
def validate_docker_installation():
    """
    Verify if Docker is installed and running on host machine.
    """
    if shutil.which('docker') is None:
        raise ExecutionException('Could not find Docker executable. Ensure Docker is installed as per the instructions at https://docs.docker.com/install/overview/.')
    cmd = ['docker', 'info']
    prc = process._exec_cmd(cmd, throw_on_error=False, capture_output=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if prc.returncode != 0:
        joined_cmd = ' '.join(cmd)
        raise ExecutionException(f'Ran `{joined_cmd}` to ensure docker daemon is running but it failed with the following output:\n{prc.stdout}')