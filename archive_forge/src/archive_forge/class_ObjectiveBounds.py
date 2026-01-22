import dataclasses
import datetime
import enum
from typing import Dict, Iterable, List, Optional, overload
from ortools.gscip import gscip_pb2
from ortools.math_opt import result_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.solvers import osqp_pb2
@dataclasses.dataclass(frozen=True)
class ObjectiveBounds:
    """Bounds on the optimal objective value.

  MOE:begin_intracomment_strip
  See go/mathopt-objective-bounds for more details.
  MOE:end_intracomment_strip

  Attributes:
    primal_bound: Solver claims there exists a primal solution that is
      numerically feasible (i.e. feasible up to the solvers tolerance), and
      whose objective value is primal_bound.

      The optimal value is equal or better (smaller for min objectives and
      larger for max objectives) than primal_bound, but only up to
      solver-tolerances.

      MOE:begin_intracomment_strip
      See go/mathopt-objective-bounds for more details.
      MOE:end_intracomment_strip
    dual_bound: Solver claims there exists a dual solution that is numerically
      feasible (i.e. feasible up to the solvers tolerance), and whose objective
      value is dual_bound.

      For MIP solvers, the associated dual problem may be some continuous
      relaxation (e.g. LP relaxation), but it is often an implicitly defined
      problem that is a complex consequence of the solvers execution. For both
      continuous and MIP solvers, the optimal value is equal or worse (larger
      for min objective and smaller for max objectives) than dual_bound, but
      only up to solver-tolerances. Some continuous solvers provide a
      numerically safer dual bound through solver's specific output (e.g. for
      PDLP, pdlp_output.convergence_information.corrected_dual_objective).

      MOE:begin_intracomment_strip
      See go/mathopt-objective-bounds for more details.
      MOE:end_intracomment_strip
  """
    primal_bound: float = 0.0
    dual_bound: float = 0.0

    def to_proto(self) -> result_pb2.ObjectiveBoundsProto:
        """Returns an equivalent proto for objective bounds."""
        return result_pb2.ObjectiveBoundsProto(primal_bound=self.primal_bound, dual_bound=self.dual_bound)