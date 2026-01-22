import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
@dataclasses.dataclass
class PrimalRay:
    """A direction of unbounded objective improvement in an optimization Model.

    Equivalently, a certificate of infeasibility for the dual of the optimization
    problem.

    E.g. consider a simple linear program:
      min c * x
      s.t. A * x >= b
      x >= 0.
    A primal ray is an x that satisfies:
      c * x < 0
      A * x >= 0
      x >= 0.
    Observe that given a feasible solution, any positive multiple of the primal
    ray plus that solution is still feasible, and gives a better objective
    value. A primal ray also proves the dual optimization problem infeasible.

    In the class PrimalRay, variable_values is this x.

    For the general case of a MathOpt optimization model, see
    go/mathopt-solutions for details.

    Attributes:
      variable_values: The value assigned for each Variable in the model.
    """
    variable_values: Dict[model.Variable, float] = dataclasses.field(default_factory=dict)