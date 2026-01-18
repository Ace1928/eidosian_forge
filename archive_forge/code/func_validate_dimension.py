import abc
import functools
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv
@staticmethod
def validate_dimension(dimension: int) -> None:
    """Raises an exception if `dimension` is not positive.

        Raises:
            ValueError: `dimension` is not positive.
        """
    if dimension < 1:
        raise ValueError(f'Wrong qid dimension. Expected a positive integer but got {dimension}.')