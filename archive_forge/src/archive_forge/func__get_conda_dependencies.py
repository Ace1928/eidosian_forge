import json
import logging
import os
import yaml
from mlflow.environment_variables import MLFLOW_CONDA_CREATE_ENV_CMD, MLFLOW_CONDA_HOME
from mlflow.exceptions import ExecutionException
from mlflow.utils import insecure_hash, process
from mlflow.utils.environment import Environment
from mlflow.utils.os import is_windows
def _get_conda_dependencies(conda_yaml_path):
    """Extracts conda dependencies from a conda yaml file.

    Args:
        conda_yaml_path: Conda yaml file path.
    """
    with open(conda_yaml_path) as f:
        conda_yaml = yaml.safe_load(f)
        return [d for d in conda_yaml.get('dependencies', []) if isinstance(d, str)]