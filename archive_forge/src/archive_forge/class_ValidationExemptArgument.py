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
class ValidationExemptArgument(typing.NamedTuple):
    """
    A NamedTuple representing the properties of an argument that is exempt from validation

    autologging_integration: The name of the autologging integration.
    function_name: The name of the function that is being validated.
    type_function: A Callable that accepts an object and returns True if the given object matches
                   the argument type. Returns False otherwise.
    positional_argument_index: The index of the argument in the function signature.
    keyword_argument_name: The name of the argument in the function signature.
    """
    autologging_integration: str
    function_name: str
    type_function: typing.Callable
    positional_argument_index: int = None
    keyword_argument_name: str = None

    def matches(self, autologging_integration, function_name, value, argument_index=None, argument_name=None):
        """
        This method checks if the properties provided through the function arguments matches the
        properties defined in the NamedTuple.

        Args:
            autologging_integration: The name of an autologging integration.
            function_name: The name of the function that is being matched.
            value: The value of the argument.
            argument_index: The index of the argument, if it is passed as a positional
                argument. Otherwise it is None.
            argument_name: The name of the argument, if it is passed as a keyword
                argument. Otherwise it is None.

        Returns:
            Returns True if the given function properties matches the exempt argument's
            properties. Returns False otherwise.
        """
        return self.autologging_integration == autologging_integration and self.function_name == function_name and (self.positional_argument_index == argument_index or self.keyword_argument_name == argument_name) and self.type_function(value)