import abc
import collections
import dataclasses
import math
import typing
from typing import (
import weakref
import immutabledict
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt.python import hash_model_storage
from ortools.math_opt.python import model_storage
class LinearProduct(LinearBase):
    """A deferred multiplication computation for linear expressions.

    This class is immutable.
    """
    __slots__ = ('_scalar', '_linear')

    def __init__(self, scalar: float, linear: LinearBase) -> None:
        if not isinstance(scalar, (float, int)):
            raise TypeError(f'unsupported type for scalar argument in LinearProduct: {type(scalar).__name__!r}')
        if not isinstance(linear, LinearBase):
            raise TypeError(f'unsupported type for linear argument in LinearProduct: {type(linear).__name__!r}')
        self._scalar: float = float(scalar)
        self._linear: LinearBase = linear

    @property
    def scalar(self) -> float:
        return self._scalar

    @property
    def linear(self) -> LinearBase:
        return self._linear

    def _flatten_once_and_add_to(self, scale: float, processed_elements: _ProcessedElements, target_stack: _ToProcessElements) -> None:
        target_stack.append(self._linear, self._scalar * scale)

    def __str__(self):
        return str(as_flat_linear_expression(self))

    def __repr__(self):
        result = f'LinearProduct({self._scalar!r}, '
        result += f'{self._linear!r})'
        return result