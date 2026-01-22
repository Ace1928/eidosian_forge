import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import List
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version
from mlflow.environment_variables import _MLFLOW_TESTING, MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import PYTHON_VERSION, insecure_hash
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd
from mlflow.utils.requirements_utils import (
from mlflow.utils.timeout import MlflowTimeoutError, run_with_timeout
from mlflow.version import VERSION
def _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements):
    """
    Validates that only one or none of `conda_env`, `pip_requirements`, and
    `extra_pip_requirements` is specified.
    """
    args = [conda_env, pip_requirements, extra_pip_requirements]
    specified = [arg for arg in args if arg is not None]
    if len(specified) > 1:
        raise ValueError('Only one of `conda_env`, `pip_requirements`, and `extra_pip_requirements` can be specified')