import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
def _compare_regular_values_equal(self, actual: torch.Tensor, expected: torch.Tensor, *, equal_nan: bool=False, identifier: Optional[Union[str, Callable[[str], str]]]=None) -> None:
    """Checks if the values of two tensors are equal."""
    self._compare_regular_values_close(actual, expected, rtol=0, atol=0, equal_nan=equal_nan, identifier=identifier)