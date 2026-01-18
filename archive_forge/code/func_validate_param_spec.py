import contextlib
import inspect
import logging
import time
from typing import List
import mlflow
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils.validation import MAX_METRICS_PER_BATCH
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient  # noqa: F401
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.autologging_utils.safety import (  # noqa: F401
from mlflow.utils.autologging_utils.versioning import (
def validate_param_spec(param_spec):
    if 'disable' not in param_spec or param_spec['disable'].default is not False:
        raise Exception(f"Invalid `autolog()` function for integration '{name}'. `autolog()` functions must specify a 'disable' argument with default value 'False'")
    elif 'silent' not in param_spec or param_spec['silent'].default is not False:
        raise Exception(f"Invalid `autolog()` function for integration '{name}'. `autolog()` functions must specify a 'silent' argument with default value 'False'")