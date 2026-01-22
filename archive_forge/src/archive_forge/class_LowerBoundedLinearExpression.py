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
class LowerBoundedLinearExpression:
    """An inequality of the form expression >= lower_bound.

    Where:
     * expression is a linear expression, and
     * lower_bound is a float
    """
    __slots__ = ('_expression', '_lower_bound')

    def __init__(self, expression: 'LinearBase', lower_bound: float) -> None:
        """Operator overloading can be used instead: e.g. `x + y >= 2.0`."""
        self._expression: 'LinearBase' = expression
        self._lower_bound: float = lower_bound

    @property
    def expression(self) -> 'LinearBase':
        return self._expression

    @property
    def lower_bound(self) -> float:
        return self._lower_bound

    def __le__(self, rhs: float) -> 'BoundedLinearExpression':
        if isinstance(rhs, (int, float)):
            return BoundedLinearExpression(self.lower_bound, self.expression, rhs)
        _raise_binary_operator_type_error('<=', type(self), type(rhs))

    def __bool__(self) -> bool:
        raise TypeError('__bool__ is unsupported for LowerBoundedLinearExpression' + '\n' + _CHAINED_COMPARISON_MESSAGE)

    def __str__(self):
        return f'{self._expression!s} >= {self._lower_bound}'

    def __repr__(self):
        return f'{self._expression!r} >= {self._lower_bound}'