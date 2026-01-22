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
def _get_Xy(args, kwargs, X_var_name, y_var_name):
    if len(args) >= 2:
        return args[:2]
    if len(args) == 1:
        return (args[0], kwargs.get(y_var_name))
    return (kwargs[X_var_name], kwargs.get(y_var_name))