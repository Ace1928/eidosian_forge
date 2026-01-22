import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from packaging.version import Version
import mlflow
from mlflow.environment_variables import MLFLOW_ENV_ROOT
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.utils.conda import _PIP_CACHE_DIR
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.environment import (
from mlflow.utils.file_utils import remove_on_error
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd, _join_commands
from mlflow.utils.requirements_utils import _parse_requirements
def _install_python(version, pyenv_root=None, capture_output=False):
    """Installs a specified version of python with pyenv and returns a path to the installed python
    binary.

    Args:
        version: Python version to install.
        pyenv_root: The value of the "PYENV_ROOT" environment variable used when running
            `pyenv install` which installs python in `{PYENV_ROOT}/versions/{version}`.
        capture_output: Set the `capture_output` argument when calling `_exec_cmd`.

    Returns:
        Path to the installed python binary.
    """
    version = version if _SEMANTIC_VERSION_REGEX.match(version) else _find_latest_installable_python_version(version)
    _logger.info('Installing python %s if it does not exist', version)
    pyenv_install_options = ('--skip-existing',) if not is_windows() else ()
    extra_env = {'PYENV_ROOT': pyenv_root} if pyenv_root else None
    pyenv_bin_path = _get_pyenv_bin_path()
    _exec_cmd([pyenv_bin_path, 'install', *pyenv_install_options, version], capture_output=capture_output, shell=is_windows(), extra_env=extra_env)
    if not is_windows():
        if pyenv_root is None:
            pyenv_root = _exec_cmd([pyenv_bin_path, 'root'], capture_output=True).stdout.strip()
        path_to_bin = ('bin', 'python')
    else:
        pyenv_root = os.getenv('PYENV_ROOT')
        if pyenv_root is None:
            raise MlflowException("Environment variable 'PYENV_ROOT' must be set")
        path_to_bin = ('python.exe',)
    return Path(pyenv_root).joinpath('versions', version, *path_to_bin)