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
@functools.wraps(original)
def _load_pyfunc_patch(*args, **kwargs):
    with cap_cm:
        model = original(*args, **kwargs)
        if input_example is not None:
            try:
                model.predict(input_example, params=params)
            except Exception as e:
                if error_file:
                    stack_trace = get_stacktrace(e)
                    write_to(error_file, 'Failed to run predict on input_example, dependencies introduced in predict are not captured.\n' + stack_trace)
                else:
                    raise e
        return model