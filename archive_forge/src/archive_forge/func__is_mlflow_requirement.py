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
def _is_mlflow_requirement(requirement_string):
    """
    Returns True if `requirement_string` represents a requirement for mlflow (e.g. 'mlflow==1.2.3').
    """
    if _MLFLOW_TESTING.get() and requirement_string == '/opt/mlflow':
        return True
    try:
        return Requirement(requirement_string).name.lower() in ['mlflow', 'mlflow-skinny']
    except InvalidRequirement:
        requirement_specifier = _get_pip_requirement_specifier(requirement_string)
        try:
            return Requirement(requirement_specifier).name.lower() == 'mlflow'
        except InvalidRequirement:
            repository_matches = ['/mlflow', 'mlflow@git']
            return any((match in requirement_string.replace(' ', '').lower() for match in repository_matches))