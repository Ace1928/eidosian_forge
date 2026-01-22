import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
@enum.unique
class SolutionStatus(enum.Enum):
    """Feasibility of a primal or dual solution as claimed by the solver.

    Attributes:
      UNDETERMINED: Solver does not claim a feasibility status.
      FEASIBLE: Solver claims the solution is feasible.
      INFEASIBLE: Solver claims the solution is infeasible.
    """
    UNDETERMINED = solution_pb2.SOLUTION_STATUS_UNDETERMINED
    FEASIBLE = solution_pb2.SOLUTION_STATUS_FEASIBLE
    INFEASIBLE = solution_pb2.SOLUTION_STATUS_INFEASIBLE