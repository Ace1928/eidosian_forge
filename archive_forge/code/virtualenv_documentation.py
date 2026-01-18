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
Runs a command in a specified virtualenv environment.

    Args:
        activate_cmd: Command to activate the virtualenv environment.
        command: Command to run in the virtualenv environment.
        install_mlflow: Flag to determine whether to install mlflow in the virtualenv
            environment.
        command_env: Environment variables passed to a process running the command.
        synchronous: Set the `synchronous` argument when calling `_exec_cmd`.
        capture_output: Set the `capture_output` argument when calling `_exec_cmd`.
        env_root_dir: See doc of PyFuncBackend constructor argument `env_root_dir`.
        kwargs: Set the `kwargs` argument when calling `_exec_cmd`.

    