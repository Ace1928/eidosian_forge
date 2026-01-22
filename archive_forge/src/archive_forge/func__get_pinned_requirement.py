import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import namedtuple
from itertools import chain, filterfalse
from pathlib import Path
from threading import Timer
from typing import List, NamedTuple, Optional
import importlib_metadata
import pkg_resources  # noqa: TID251
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version
import mlflow
from mlflow.environment_variables import MLFLOW_REQUIREMENTS_INFERENCE_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils.versioning import _strip_dev_version_suffix
from mlflow.utils.databricks_utils import (
def _get_pinned_requirement(req_str, version=None, module=None):
    """Returns a string representing a pinned pip requirement to install the specified package and
    version (e.g. 'mlflow==1.2.3').

    Args:
        req_str: The package requirement string (e.g. "mlflow" or "mlflow[gateway]").
        version: The version of the package. If None, defaults to the installed version.
        module: The name of the top-level module provided by the package . For example,
            if `package` is 'scikit-learn', `module` should be 'sklearn'. If None, defaults
            to `package`.
        extras: A list of extra names for the package.

    """
    req = Requirement(req_str)
    package = req.name
    if version is None:
        version_raw = _get_installed_version(package, module)
        local_version_label = _get_local_version_label(version_raw)
        if local_version_label:
            version = _strip_local_version_label(version_raw)
            if not (is_in_databricks_runtime() and package in ('torch', 'torchvision')):
                msg = f"Found {package} version ({version_raw}) contains a local version label (+{local_version_label}). MLflow logged a pip requirement for this package as '{package}=={version}' without the local version label to make it installable from PyPI. To specify pip requirements containing local version labels, please use `conda_env` or `pip_requirements`."
                _logger.warning(msg)
        else:
            version = version_raw
    if req.extras:
        return f'{package}[{','.join(req.extras)}]=={version}'
    return f'{package}=={version}'