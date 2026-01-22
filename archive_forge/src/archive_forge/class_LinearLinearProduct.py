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
class LinearLinearProduct(QuadraticBase):
    """A deferred multiplication of two linear expressions.

    This class is immutable.
    """
    __slots__ = ('_first_linear', '_second_linear')

    def __init__(self, first_linear: LinearBase, second_linear: LinearBase) -> None:
        if not isinstance(first_linear, LinearBase):
            raise TypeError(f'unsupported type for first_linear argument in LinearLinearProduct: {type(first_linear).__name__!r}')
        if not isinstance(second_linear, LinearBase):
            raise TypeError(f'unsupported type for second_linear argument in LinearLinearProduct: {type(second_linear).__name__!r}')
        self._first_linear: LinearBase = first_linear
        self._second_linear: LinearBase = second_linear

    @property
    def first_linear(self) -> LinearBase:
        return self._first_linear

    @property
    def second_linear(self) -> LinearBase:
        return self._second_linear

    def _quadratic_flatten_once_and_add_to(self, scale: float, processed_elements: _QuadraticProcessedElements, target_stack: _QuadraticToProcessElements) -> None:
        first_expression = as_flat_linear_expression(self._first_linear)
        second_expression = as_flat_linear_expression(self._second_linear)
        processed_elements.offset += first_expression.offset * second_expression.offset * scale
        for first_var, first_val in first_expression.terms.items():
            processed_elements.terms[first_var] += second_expression.offset * first_val * scale
        for second_var, second_val in second_expression.terms.items():
            processed_elements.terms[second_var] += first_expression.offset * second_val * scale
        for first_var, first_val in first_expression.terms.items():
            for second_var, second_val in second_expression.terms.items():
                processed_elements.quadratic_terms[QuadraticTermKey(first_var, second_var)] += first_val * second_val * scale

    def __str__(self):
        return str(as_flat_quadratic_expression(self))

    def __repr__(self):
        result = 'LinearLinearProduct('
        result += f'{self._first_linear!r}, '
        result += f'{self._second_linear!r})'
        return result