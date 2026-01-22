import collections
import inspect
import logging
import pkgutil
import platform
import warnings
from copy import deepcopy
from importlib import import_module
from numbers import Number
from operator import itemgetter
import numpy as np
from packaging.version import Version
from mlflow import MlflowClient
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.file_utils import TempDir
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.time import get_current_time_millis
def _log_warning_for_artifacts(func_name, func_call, err):
    msg = func_call.__qualname__ + ' failed. The artifact ' + func_name + ' will not be recorded.' + ' Artifact error: ' + str(err)
    _logger.warning(msg)