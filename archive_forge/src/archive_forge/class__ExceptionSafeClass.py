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
class _ExceptionSafeClass(base_class):
    """
        Metaclass that wraps all functions defined on the specified class with broad error handling
        logic to guard against unexpected errors during autlogging.

        Rationale: Patched autologging functions commonly pass additional class instances as
        arguments to their underlying original training routines; for example, Keras autologging
        constructs a subclass of `keras.callbacks.Callback` and forwards it to `Model.fit()`.
        To prevent errors encountered during method execution within such classes from disrupting
        model training, this metaclass wraps all class functions in a broad try / catch statement.

        Note: `ExceptionSafeClass` does not handle exceptions in class methods or static methods,
        as these are not always Python callables and are difficult to wrap
        """

    def __new__(cls, name, bases, dct):
        for m in dct:
            if callable(dct[m]):
                dct[m] = exception_safe_function_for_class(dct[m])
        return base_class.__new__(cls, name, bases, dct)