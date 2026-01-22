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