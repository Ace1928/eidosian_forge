import importlib
import re
from packaging.version import InvalidVersion, Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.databricks_utils import is_in_databricks_runtime
def _check_spark_version_in_range(ver, min_ver, max_ver):
    """
    Utility function for allowing late addition release changes to PySpark minor version increments
    to be accepted, provided that the previous minor version has been previously validated.
    For example, if version 3.2.1 has been validated as functional with MLflow, an upgrade of
    PySpark's minor version to 3.2.2 will still provide a valid version check.
    """
    parsed_ver = Version(ver)
    if parsed_ver > Version(min_ver):
        ver = f'{parsed_ver.major}.{parsed_ver.minor}'
    return _check_version_in_range(ver, min_ver, max_ver)