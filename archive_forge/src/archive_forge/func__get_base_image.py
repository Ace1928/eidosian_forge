import ctypes
import logging
import os
import pathlib
import posixpath
import shlex
import signal
import subprocess
import sys
import warnings
from pathlib import Path
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import FlavorBackend, docker_utils
from mlflow.models.docker_utils import PYTHON_SLIM_BASE_IMAGE, UBUNTU_BASE_IMAGE
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.pyfunc import (
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.conda import get_conda_bin_executable, get_or_create_conda_env
from mlflow.utils.environment import Environment, _PythonEnv
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import _get_all_flavor_configurations
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.os import is_windows
from mlflow.utils.process import ShellCommandException, cache_return_value_per_process
from mlflow.utils.virtualenv import (
from mlflow.version import VERSION
def _get_base_image(self, model_path: str, install_java: bool) -> str:
    """
        Determine the base image to use for the Dockerfile.

        We use Python slim base image when all of the following conditions are met:
          1. Model URI is specified by the user
          2. Model flavor does not require Java
          3. Python version is specified in the model

        Returns:
            Either the Ubuntu base image or the Python slim base image.
        """
    if not install_java:
        flavors = _get_all_flavor_configurations(model_path).keys()
        if (java_flavors := (JAVA_FLAVORS & flavors)):
            _logger.info(f'Detected java flavors {java_flavors}, installing Java in the image')
            install_java = True
    if install_java:
        return UBUNTU_BASE_IMAGE
    try:
        model_config_path = os.path.join(model_path, MLMODEL_FILE_NAME)
        model = Model.load(model_config_path)
        conf = model.flavors[pyfunc.FLAVOR_NAME]
        env_conf = conf[pyfunc.ENV]
        python_env_config_path = os.path.join(model_path, env_conf[_EnvManager.VIRTUALENV])
        python_env = _PythonEnv.from_yaml(python_env_config_path)
        return PYTHON_SLIM_BASE_IMAGE.format(version=python_env.python)
    except Exception as e:
        _logger.warning(f'Failed to determine Python version from {model_config_path}. Defaulting to {UBUNTU_BASE_IMAGE}. Error: {e}')
        return UBUNTU_BASE_IMAGE