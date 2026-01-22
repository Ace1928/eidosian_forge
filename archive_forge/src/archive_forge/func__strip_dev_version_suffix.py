import importlib
import re
from packaging.version import InvalidVersion, Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.databricks_utils import is_in_databricks_runtime
def _strip_dev_version_suffix(version):
    return re.sub('(\\.?)dev.*', '', version)