import dataclasses
from typing import Mapping
import immutabledict
from ortools.math_opt import infeasible_subsystem_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import result
@dataclasses.dataclass(frozen=True)
class ModelSubset:
    """A subset of a Model's constraints (including variable bounds/integrality).

    When returned from `solve.compute_infeasible_subsystem`, the contained
    `ModelSubsetBounds` will all be nonempty.

    Attributes:
      variable_bounds: The upper and/or lower bound constraints on these variables
        are included in the subset.
      variable_integrality: The constraint that a variable is integer is included
        in the subset.
      linear_constraints: The upper and/or lower bounds from these linear
        constraints are included in the subset.
    """
    variable_bounds: Mapping[model.Variable, ModelSubsetBounds] = immutabledict.immutabledict()
    variable_integrality: frozenset[model.Variable] = frozenset()
    linear_constraints: Mapping[model.LinearConstraint, ModelSubsetBounds] = immutabledict.immutabledict()

    def empty(self) -> bool:
        """Returns true if all the nested constraint collections are empty.

        Warning: When `self.variable_bounds` or `self.linear_constraints` contain
        only ModelSubsetBounds which are themselves empty, this function will return
        False.

        Returns:
          True if this is empty.
        """
        return not (self.variable_bounds or self.variable_integrality or self.linear_constraints)

    def to_proto(self) -> infeasible_subsystem_pb2.ModelSubsetProto:
        """Returns an equivalent proto message for this `ModelSubset`."""
        return infeasible_subsystem_pb2.ModelSubsetProto(variable_bounds={var.id: bounds.to_proto() for var, bounds in self.variable_bounds.items()}, variable_integrality=sorted((var.id for var in self.variable_integrality)), linear_constraints={con.id: bounds.to_proto() for con, bounds in self.linear_constraints.items()})