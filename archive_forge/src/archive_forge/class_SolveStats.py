import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
@dataclasses.dataclass
class SolveStats:
    """Problem statuses and solve statistics returned by the solver.

    Attributes:
      solve_time: Elapsed wall clock time as measured by math_opt, roughly the
        time inside solve(). Note: this does not include work done building the
        model.
      simplex_iterations: Simplex iterations.
      barrier_iterations: Barrier iterations.
      first_order_iterations: First order iterations.
      node_count: Node count.
    """
    solve_time: datetime.timedelta = datetime.timedelta()
    simplex_iterations: int = 0
    barrier_iterations: int = 0
    first_order_iterations: int = 0
    node_count: int = 0

    def to_proto(self) -> result_pb2.SolveStatsProto:
        """Returns an equivalent proto for a solve stats."""
        result = result_pb2.SolveStatsProto(simplex_iterations=self.simplex_iterations, barrier_iterations=self.barrier_iterations, first_order_iterations=self.first_order_iterations, node_count=self.node_count)
        result.solve_time.FromTimedelta(self.solve_time)
        return result