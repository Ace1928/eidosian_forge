import abc
import dataclasses
from typing import Iterator, Optional, Type, TypeVar
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
class BadLinearConstraintIdError(LookupError):
    """Raised by ModelStorage when a bad linear constraint id is given."""

    def __init__(self, linear_constraint_id):
        super().__init__(f'Unexpected linear constraint id: {linear_constraint_id}')
        self.id = linear_constraint_id