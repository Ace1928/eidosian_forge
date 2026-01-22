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
def as_quadratic_expression(self) -> QuadraticExpression:
    return as_flat_quadratic_expression(self.offset + LinearSum(self.linear_terms()) + QuadraticSum(self.quadratic_terms()))