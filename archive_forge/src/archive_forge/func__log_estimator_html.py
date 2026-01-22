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
def _log_estimator_html(run_id, estimator):
    if not _is_estimator_html_repr_supported():
        return
    from sklearn.utils import estimator_html_repr
    estimator_html_string = f'\n<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8"/>\n  </head>\n  <body>\n    {estimator_html_repr(estimator)}\n  </body>\n</html>\n    '
    MlflowClient().log_text(run_id, estimator_html_string, artifact_file='estimator.html')