import dataclasses
from typing import Dict, List, Optional
from ortools.math_opt import model_parameters_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.python import sparse_containers
@dataclasses.dataclass
class ModelSolveParameters:
    """Model specific solver configuration, for example, an initial basis.

    This class mirrors (and can generate) the related proto
    model_parameters_pb2.ModelSolveParametersProto.

    Attributes:
      variable_values_filter: Only return solution and primal ray values for
        variables accepted by this filter (default accepts all variables).
      dual_values_filter: Only return dual variable values and dual ray values for
        linear constraints accepted by thei filter (default accepts all linear
        constraints).
      reduced_costs_filter: Only return reduced cost and dual ray values for
        variables accepted by this filter (default accepts all variables).
      initial_basis: If set, provides a warm start for simplex based solvers.
      solution_hints: Optional solution hints. If the underlying solver only
        accepts a single hint, the first hint is used.
      branching_priorities: Optional branching priorities. Variables with higher
        values will be branched on first. Variables for which priorities are not
        set get the solver's default priority (usually zero).
    """
    variable_values_filter: sparse_containers.VariableFilter = sparse_containers.VariableFilter()
    dual_values_filter: sparse_containers.LinearConstraintFilter = sparse_containers.LinearConstraintFilter()
    reduced_costs_filter: sparse_containers.VariableFilter = sparse_containers.VariableFilter()
    initial_basis: Optional[solution.Basis] = None
    solution_hints: List[SolutionHint] = dataclasses.field(default_factory=list)
    branching_priorities: Dict[model.Variable, int] = dataclasses.field(default_factory=dict)

    def to_proto(self) -> model_parameters_pb2.ModelSolveParametersProto:
        """Returns an equivalent protocol buffer."""
        result = model_parameters_pb2.ModelSolveParametersProto(variable_values_filter=self.variable_values_filter.to_proto(), dual_values_filter=self.dual_values_filter.to_proto(), reduced_costs_filter=self.reduced_costs_filter.to_proto(), branching_priorities=sparse_containers.to_sparse_int32_vector_proto(self.branching_priorities))
        if self.initial_basis:
            result.initial_basis.CopyFrom(self.initial_basis.to_proto())
        for hint in self.solution_hints:
            result.solution_hints.append(hint.to_proto())
        return result