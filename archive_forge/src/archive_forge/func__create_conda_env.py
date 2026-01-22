import json
import logging
import os
import yaml
from mlflow.environment_variables import MLFLOW_CONDA_CREATE_ENV_CMD, MLFLOW_CONDA_HOME
from mlflow.exceptions import ExecutionException
from mlflow.utils import insecure_hash, process
from mlflow.utils.environment import Environment
from mlflow.utils.os import is_windows
def _create_conda_env(conda_env_path, conda_env_create_path, project_env_name, conda_extra_env_vars, capture_output):
    if conda_env_path:
        process._exec_cmd([conda_env_create_path, 'env', 'create', '-n', project_env_name, '--file', conda_env_path, '--quiet'], extra_env=conda_extra_env_vars, capture_output=capture_output)
    else:
        process._exec_cmd([conda_env_create_path, 'create', '--channel', 'conda-forge', '--yes', '--override-channels', '-n', project_env_name, 'python'], extra_env=conda_extra_env_vars, capture_output=capture_output)
    return Environment(get_conda_command(project_env_name), conda_extra_env_vars)