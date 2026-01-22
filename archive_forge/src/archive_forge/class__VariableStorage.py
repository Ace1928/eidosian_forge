from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
import weakref
from ortools.math_opt import model_pb2
from ortools.math_opt import model_update_pb2
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model_storage
class _VariableStorage:
    """Data specific to each decision variable in the optimization problem."""

    def __init__(self, lb: float, ub: float, is_integer: bool, name: str) -> None:
        self.lower_bound: float = lb
        self.upper_bound: float = ub
        self.is_integer: bool = is_integer
        self.name: str = name
        self.linear_constraint_nonzeros: Set[int] = set()