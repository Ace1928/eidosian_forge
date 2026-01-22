import json
import logging
import os
import yaml
from mlflow.environment_variables import MLFLOW_CONDA_CREATE_ENV_CMD, MLFLOW_CONDA_HOME
from mlflow.exceptions import ExecutionException
from mlflow.utils import insecure_hash, process
from mlflow.utils.environment import Environment
from mlflow.utils.os import is_windows
def _get_conda_extra_env_vars(env_root_dir=None):
    """
    Given the `env_root_dir` (See doc of PyFuncBackend constructor argument `env_root_dir`),
    return a dict of environment variables which are used to config conda to generate envs
    under the expected `env_root_dir`.
    """
    if env_root_dir is None:
        return None
    conda_envs_path = os.path.join(env_root_dir, _CONDA_ENVS_DIR)
    conda_pkgs_path = os.path.join(env_root_dir, _CONDA_CACHE_PKGS_DIR)
    pip_cache_dir = os.path.join(env_root_dir, _PIP_CACHE_DIR)
    os.makedirs(conda_envs_path, exist_ok=True)
    os.makedirs(conda_pkgs_path, exist_ok=True)
    os.makedirs(pip_cache_dir, exist_ok=True)
    return {'CONDA_ENVS_PATH': conda_envs_path, 'CONDA_PKGS_DIRS': conda_pkgs_path, 'PIP_CACHE_DIR': pip_cache_dir, 'PIP_NO_INPUT': '1'}