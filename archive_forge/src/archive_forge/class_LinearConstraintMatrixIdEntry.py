import abc
import dataclasses
from typing import Iterator, Optional, Type, TypeVar
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
@dataclasses.dataclass(frozen=True)
class LinearConstraintMatrixIdEntry:
    __slots__ = ('linear_constraint_id', 'variable_id', 'coefficient')
    linear_constraint_id: int
    variable_id: int
    coefficient: float