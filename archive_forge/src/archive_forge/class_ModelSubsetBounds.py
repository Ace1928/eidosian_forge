import dataclasses
from typing import Mapping
import immutabledict
from ortools.math_opt import infeasible_subsystem_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import result
@dataclasses.dataclass(frozen=True)
class ModelSubsetBounds:
    """Presence of the upper and lower bounds in a two-sided constraint.

    E.g. for 1 <= x <= 2, `lower` is the constraint 1 <= x and `upper` is the
    constraint x <= 2.

    Attributes:
      lower: If the lower bound half of the two-sided constraint is selected.
      upper: If the upper bound half of the two-sided constraint is selected.
    """
    lower: bool = False
    upper: bool = False

    def empty(self) -> bool:
        """Is empty if both `lower` and `upper` are False."""
        return not (self.lower or self.upper)

    def to_proto(self) -> infeasible_subsystem_pb2.ModelSubsetProto.Bounds:
        """Returns an equivalent proto message for these bounds."""
        return infeasible_subsystem_pb2.ModelSubsetProto.Bounds(lower=self.lower, upper=self.upper)