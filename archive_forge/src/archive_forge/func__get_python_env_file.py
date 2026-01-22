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
def _get_python_env_file(model_config):
    from mlflow.pyfunc import EnvType
    for flavor, config in model_config.flavors.items():
        if flavor == mlflow.pyfunc.FLAVOR_NAME:
            env = config.get(mlflow.pyfunc.ENV)
            if isinstance(env, dict):
                return env[EnvType.VIRTUALENV]
    return _PYTHON_ENV_FILE_NAME