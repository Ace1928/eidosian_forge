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
def _get_or_create_virtualenv(local_model_path, env_id=None, env_root_dir=None, capture_output=False, pip_requirements_override=None):
    """Restores an MLflow model's environment with pyenv and virtualenv and returns a command
    to activate it.

    Args:
        local_model_path: Local directory containing the model artifacts.
        env_id: Optional string that is added to the contents of the yaml file before
            calculating the hash. It can be used to distinguish environments that have the
            same conda dependencies but are supposed to be different based on the context.
            For example, when serving the model we may install additional dependencies to the
            environment after the environment has been activated.
        pip_requirements_override: If specified, install the specified python dependencies to
            the environment (upgrade if already installed).

    Returns:
        Command to activate the created virtualenv environment
        (e.g. "source /path/to/bin/activate").

    """
    _validate_pyenv_is_available()
    _validate_virtualenv_is_available()
    local_model_path = Path(local_model_path)
    python_env = _get_python_env(local_model_path)
    extra_env = _get_virtualenv_extra_env_vars(env_root_dir)
    if env_root_dir is not None:
        virtual_envs_root_path = Path(env_root_dir) / _VIRTUALENV_ENVS_DIR
        pyenv_root_path = Path(env_root_dir) / _PYENV_ROOT_DIR
        pyenv_root_path.mkdir(parents=True, exist_ok=True)
        pyenv_root_dir = str(pyenv_root_path)
    else:
        virtual_envs_root_path = Path(_get_mlflow_virtualenv_root())
        pyenv_root_dir = None
    virtual_envs_root_path.mkdir(parents=True, exist_ok=True)
    python_bin_path = _install_python(python_env.python, pyenv_root=pyenv_root_dir, capture_output=capture_output)
    env_name = _get_virtualenv_name(python_env, local_model_path, env_id)
    env_dir = virtual_envs_root_path / env_name
    try:
        activate_cmd = _create_virtualenv(local_model_path, python_bin_path, env_dir, python_env, extra_env=extra_env, capture_output=capture_output)
        if pip_requirements_override:
            _logger.info(f'Installing additional dependencies specified by pip_requirements_override: {pip_requirements_override}')
            cmd = _join_commands(activate_cmd, f'python -m pip install --quiet -U {' '.join(pip_requirements_override)}')
            _exec_cmd(cmd, capture_output=capture_output, extra_env=extra_env)
        return activate_cmd
    except:
        _logger.warning('Encountered unexpected error while creating %s', env_dir)
        if env_dir.exists():
            _logger.warning('Attempting to remove %s', env_dir)
            shutil.rmtree(env_dir, ignore_errors=True)
            msg = 'Failed to remove %s' if env_dir.exists() else 'Successfully removed %s'
            _logger.warning(msg, env_dir)
        raise