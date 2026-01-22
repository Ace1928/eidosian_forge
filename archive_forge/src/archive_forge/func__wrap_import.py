import argparse
import builtins
import functools
import importlib
import json
import os
import sys
import mlflow
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.pyfunc import MAIN
from mlflow.utils._spark_utils import _prepare_subprocess_environ_for_creating_local_spark_session
from mlflow.utils.exception_utils import get_stacktrace
from mlflow.utils.file_utils import write_to
from mlflow.utils.requirements_utils import (
def _wrap_import(self, original):

    @functools.wraps(original)
    def wrapper(name, globals=None, locals=None, fromlist=(), level=0):
        is_absolute_import = level == 0
        if is_absolute_import:
            self._record_imported_module(name)
        return original(name, globals, locals, fromlist, level)
    return wrapper