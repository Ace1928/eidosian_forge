from __future__ import annotations
import atexit
import concurrent.futures
import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, overload
from langsmith import client as ls_client
from langsmith import run_helpers as rh
from langsmith import utils as ls_utils
def to_contain(self, value: Any) -> None:
    """Assert that the expectation value contains the given value.

        Args:
            value: The value to check for containment.

        Raises:
            AssertionError: If the expectation value does not contain the given value.
        """
    self._assert(value in self.value, f'Expected {self.key} to contain {value}, but it does not', 'to_contain')