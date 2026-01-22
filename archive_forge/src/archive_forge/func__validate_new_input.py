import abc
import functools
import inspect
import itertools
import typing
import uuid
from abc import abstractmethod
from contextlib import contextmanager
import mlflow
import mlflow.utils.autologging_utils
from mlflow.entities.run_status import RunStatus
from mlflow.environment_variables import _MLFLOW_AUTOLOGGING_TESTING
from mlflow.tracking.client import MlflowClient
from mlflow.utils import gorilla, is_iterator
from mlflow.utils.autologging_utils import _logger
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING
def _validate_new_input(inp):
    """
        Validates a new input (arg or kwarg) introduced to the underlying / original ML function
        call during the execution of a patched ML function. The new input is valid if:

            - The new input is a function that has been decorated with
              `exception_safe_function_for_class` or `pickalable_exception_safe_function`
            - OR the new input is a class with the `ExceptionSafeClass` metaclass
            - OR the new input is a list and each of its elements is valid according to the
              these criteria
        """
    if type(inp) == list:
        for item in inp:
            _validate_new_input(item)
    elif isinstance(inp, dict) and 'callbacks' in inp:
        _validate_new_input(inp['callbacks'])
    elif callable(inp):
        assert getattr(inp, _ATTRIBUTE_EXCEPTION_SAFE, False), f"New function argument '{inp}' passed to original function is not exception-safe. Please decorate the function with `exception_safe_function` or `pickalable_exception_safe_function`"
    else:
        assert hasattr(inp, '__class__') and type(inp.__class__) in [ExceptionSafeClass, ExceptionSafeAbstractClass], f"Invalid new input '{inp}'. New args / kwargs introduced to `original` function calls by patched code must either be functions decorated with `exception_safe_function_for_class`, instances of classes with the `ExceptionSafeClass` or `ExceptionSafeAbstractClass` metaclass safe or lists of such exception safe functions / classes."