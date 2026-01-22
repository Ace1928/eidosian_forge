import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
@enum.unique
class BasisStatus(enum.Enum):
    """Status of a variable/constraint in a LP basis.

    Attributes:
      FREE: The variable/constraint is free (it has no finite bounds).
      AT_LOWER_BOUND: The variable/constraint is at its lower bound (which must be
        finite).
      AT_UPPER_BOUND: The variable/constraint is at its upper bound (which must be
        finite).
      FIXED_VALUE: The variable/constraint has identical finite lower and upper
        bounds.
      BASIC: The variable/constraint is basic.
    """
    FREE = solution_pb2.BASIS_STATUS_FREE
    AT_LOWER_BOUND = solution_pb2.BASIS_STATUS_AT_LOWER_BOUND
    AT_UPPER_BOUND = solution_pb2.BASIS_STATUS_AT_UPPER_BOUND
    FIXED_VALUE = solution_pb2.BASIS_STATUS_FIXED_VALUE
    BASIC = solution_pb2.BASIS_STATUS_BASIC