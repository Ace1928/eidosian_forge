import dataclasses
from typing import Mapping
import immutabledict
from ortools.math_opt import infeasible_subsystem_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import result
@dataclasses.dataclass(frozen=True)
class ComputeInfeasibleSubsystemResult:
    """The result of searching for an infeasible subsystem.

    This is the result of calling `mathopt.compute_infeasible_subsystem()`.

    Attributes:
      feasibility: If the problem was proven feasible, infeasible, or no
        conclusion was reached. The fields below are ignored unless the problem
        was proven infeasible.
      infeasible_subsystem: Ignored unless `feasibility` is `INFEASIBLE`, a subset
        of the model that is still infeasible.
      is_minimal: Ignored unless `feasibility` is `INFEASIBLE`. If True, then the
        removal of any constraint from `infeasible_subsystem` makes the sub-model
        feasible. Note that, due to problem transformations MathOpt applies or
        idiosyncrasies of the solvers contract, the returned infeasible subsystem
        may not actually be minimal.
    """
    feasibility: result.FeasibilityStatus = result.FeasibilityStatus.UNDETERMINED
    infeasible_subsystem: ModelSubset = ModelSubset()
    is_minimal: bool = False

    def to_proto(self) -> infeasible_subsystem_pb2.ComputeInfeasibleSubsystemResultProto:
        """Returns an equivalent proto for this `ComputeInfeasibleSubsystemResult`."""
        return infeasible_subsystem_pb2.ComputeInfeasibleSubsystemResultProto(feasibility=self.feasibility.value, infeasible_subsystem=self.infeasible_subsystem.to_proto(), is_minimal=self.is_minimal)