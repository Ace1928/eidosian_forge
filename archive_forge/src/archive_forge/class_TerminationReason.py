import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
@enum.unique
class TerminationReason(enum.Enum):
    """The reason a solve of a model terminated.

    These reasons are typically as reported by the underlying solver, e.g. we do
    not attempt to verify the precision of the solution returned.

    The values are:
       * OPTIMAL: A provably optimal solution (up to numerical tolerances) has
           been found.
       * INFEASIBLE: The primal problem has no feasible solutions.
       * UNBOUNDED: The primal problem is feasible and arbitrarily good solutions
           can be found along a primal ray.
       * INFEASIBLE_OR_UNBOUNDED: The primal problem is either infeasible or
           unbounded. More details on the problem status may be available in
           solve_stats.problem_status. Note that Gurobi's unbounded status may be
           mapped here as explained in
           go/mathopt-solver-specific#gurobi-inf-or-unb.
       * IMPRECISE: The problem was solved to one of the criteria above (Optimal,
           Infeasible, Unbounded, or InfeasibleOrUnbounded), but one or more
           tolerances was not met. Some primal/dual solutions/rays may be present,
           but either they will be slightly infeasible, or (if the problem was
           nearly optimal) their may be a gap between the best solution objective
           and best objective bound.

           Users can still query primal/dual solutions/rays and solution stats,
           but they are responsible for dealing with the numerical imprecision.
       * FEASIBLE: The optimizer reached some kind of limit and a primal feasible
           solution is returned. See SolveResultProto.limit_detail for detailed
           description of the kind of limit that was reached.
       * NO_SOLUTION_FOUND: The optimizer reached some kind of limit and it did
           not find a primal feasible solution. See SolveResultProto.limit_detail
           for detailed description of the kind of limit that was reached.
       * NUMERICAL_ERROR: The algorithm stopped because it encountered
           unrecoverable numerical error. No solution information is present.
       * OTHER_ERROR: The algorithm stopped because of an error not covered by one
           of the statuses defined above. No solution information is present.
    """
    OPTIMAL = result_pb2.TERMINATION_REASON_OPTIMAL
    INFEASIBLE = result_pb2.TERMINATION_REASON_INFEASIBLE
    UNBOUNDED = result_pb2.TERMINATION_REASON_UNBOUNDED
    INFEASIBLE_OR_UNBOUNDED = result_pb2.TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED
    IMPRECISE = result_pb2.TERMINATION_REASON_IMPRECISE
    FEASIBLE = result_pb2.TERMINATION_REASON_FEASIBLE
    NO_SOLUTION_FOUND = result_pb2.TERMINATION_REASON_NO_SOLUTION_FOUND
    NUMERICAL_ERROR = result_pb2.TERMINATION_REASON_NUMERICAL_ERROR
    OTHER_ERROR = result_pb2.TERMINATION_REASON_OTHER_ERROR