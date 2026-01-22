import json
import logging
import os
import yaml
from mlflow.environment_variables import MLFLOW_CONDA_CREATE_ENV_CMD, MLFLOW_CONDA_HOME
from mlflow.exceptions import ExecutionException
from mlflow.utils import insecure_hash, process
from mlflow.utils.environment import Environment
from mlflow.utils.os import is_windows
def _create_conda_env_retry(conda_env_path, conda_env_create_path, project_env_name, conda_extra_env_vars, _capture_output):
    """
    `conda env create` command can fail due to network issues such as `ConnectionResetError`
    while collecting package metadata. This function retries the command up to 3 times.
    """
    num_attempts = 3
    for attempt in range(num_attempts):
        try:
            return _create_conda_env(conda_env_path, conda_env_create_path, project_env_name, conda_extra_env_vars, capture_output=True)
        except process.ShellCommandException as e:
            error_str = str(e)
            if num_attempts - attempt - 1 > 0 and ('ConnectionResetError' in error_str or 'ChunkedEncodingError' in error_str):
                _logger.warning('Conda env creation failed due to network issue. Retrying...')
                continue
            raise