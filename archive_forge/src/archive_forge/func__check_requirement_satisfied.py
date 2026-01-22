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
def _check_requirement_satisfied(requirement_str):
    """
    Checks whether the current python environment satisfies the given requirement if it is parsable
    as a package name and a set of version specifiers, and returns a `_MismatchedPackageInfo`
    object containing the mismatched package name, installed version, and requirement if the
    requirement is not satisfied. Otherwise, returns None.
    """
    _init_packages_to_modules_map()
    try:
        req = Requirement(requirement_str)
    except Exception:
        return
    pkg_name = req.name
    try:
        installed_version = _get_installed_version(pkg_name, _PACKAGES_TO_MODULES.get(pkg_name))
    except ModuleNotFoundError:
        return _MismatchedPackageInfo(package_name=pkg_name, installed_version=None, requirement=requirement_str)
    if pkg_name == 'mlflow' and 'gateway' in req.extras:
        try:
            from mlflow import gateway
        except ModuleNotFoundError:
            return _MismatchedPackageInfo(package_name='mlflow[gateway]', installed_version=None, requirement=requirement_str)
    if pkg_name == 'mlflow' and installed_version == mlflow.__version__ and Version(installed_version).is_devrelease:
        return None
    if len(req.specifier) > 0 and (not req.specifier.contains(installed_version)):
        return _MismatchedPackageInfo(package_name=pkg_name, installed_version=installed_version, requirement=requirement_str)
    return None