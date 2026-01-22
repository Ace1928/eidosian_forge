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
def _deduplicate_requirements(requirements):
    """
    De-duplicates a list of pip package requirements, handling complex scenarios such as merging
    extras and combining version constraints.

    This function processes a list of pip package requirements and de-duplicates them. It handles
    standard PyPI packages and non-standard requirements (like URLs or local paths). The function
    merges extras and combines version constraints for duplicate packages. The most restrictive
    version specifications or the ones with extras are prioritized. If incompatible version
    constraints are detected, it raises an MlflowException.

    Args:
        requirements (list of str): A list of pip package requirement strings.

    Returns:
        list of str: A deduplicated list of pip package requirements.

    Raises:
        MlflowException: If incompatible version constraints are detected among the provided
                         requirements.

    Examples:
        - Input: ["packageA", "packageA==1.0"]
          Output: ["packageA==1.0"]

        - Input: ["packageX>1.0", "packageX[extras]", "packageX<2.0"]
          Output: ["packageX[extras]>1.0,<2.0"]

        - Input: ["markdown[extra1]>=3.5.1", "markdown[extra2]<4", "markdown"]
          Output: ["markdown[extra1,extra2]>=3.5.1,<4"]

        - Input: ["scikit-learn==1.1", "scikit-learn<1"]
          Raises MlflowException indicating incompatible versions.

    Note:
        - Non-standard requirements (like URLs or file paths) are included as-is.
        - If a requirement appears multiple times with different sets of extras, they are merged.
        - The function uses `_validate_version_constraints` to check for incompatible version
          constraints by doing a dry-run pip install of a requirements collection.
    """
    deduped_reqs = {}
    for req in requirements:
        try:
            parsed_req = Requirement(req)
            base_pkg = parsed_req.name
            existing_req = deduped_reqs.get(base_pkg)
            if not existing_req:
                deduped_reqs[base_pkg] = parsed_req
            else:
                if existing_req.specifier and parsed_req.specifier and (existing_req.specifier != parsed_req.specifier):
                    _validate_version_constraints([str(existing_req), req])
                    parsed_req.specifier = ','.join([str(existing_req.specifier), str(parsed_req.specifier)])
                if existing_req.specifier and (not parsed_req.specifier):
                    parsed_req.specifier = existing_req.specifier
                if existing_req.extras and parsed_req.extras and (existing_req.extras != parsed_req.extras):
                    parsed_req.extras = sorted(set(existing_req.extras).union(parsed_req.extras))
                elif existing_req.extras and (not parsed_req.extras):
                    parsed_req.extras = existing_req.extras
                deduped_reqs[base_pkg] = parsed_req
        except InvalidRequirement:
            if req not in deduped_reqs:
                deduped_reqs[req] = req
    return [str(req) for req in deduped_reqs.values()]