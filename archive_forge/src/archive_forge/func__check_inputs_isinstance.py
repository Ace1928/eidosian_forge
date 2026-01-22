import abc
import cmath
import collections.abc
import contextlib
import warnings
from typing import (
import torch
@staticmethod
def _check_inputs_isinstance(*inputs: Any, cls: Union[Type, Tuple[Type, ...]]):
    """Checks if all inputs are instances of a given class and raise :class:`UnsupportedInputs` otherwise."""
    if not all((isinstance(input, cls) for input in inputs)):
        Pair._inputs_not_supported()