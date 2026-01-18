from __future__ import annotations
import atexit
import concurrent.futures
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, overload
from langsmith import client as ls_client
from langsmith import run_helpers as rh
from langsmith import utils as ls_utils
def to_be_between(self, min_value: float, max_value: float) -> None:
    """Assert that the expectation value is between the given min and max values.

        Args:
            min_value: The minimum value (exclusive).
            max_value: The maximum value (exclusive).

        Raises:
            AssertionError: If the expectation value
                is not between the given min and max.
        """
    self._assert(min_value < self.value < max_value, f'Expected {self.key} to be between {min_value} and {max_value}, but got {self.value}', 'to_be_between')