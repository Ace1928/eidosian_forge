import dataclasses
from typing import Dict, List, Optional
from ortools.math_opt import model_parameters_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import solution
from ortools.math_opt.python import sparse_containers
@dataclasses.dataclass
class SolutionHint:
    """A suggested starting solution for the solver.

    MIP solvers generally only want primal information (`variable_values`),
    while LP solvers want both primal and dual information (`dual_values`).

    Many MIP solvers can work with: (1) partial solutions that do not specify all
    variables or (2) infeasible solutions. In these cases, solvers typically solve
    a sub-MIP to complete/correct the hint.

    How the hint is used by the solver, if at all, is highly dependent on the
    solver, the problem type, and the algorithm used. The most reliable way to
    ensure your hint has an effect is to read the underlying solvers logs with
    and without the hint.

    Simplex-based LP solvers typically prefer an initial basis to a solution
    hint (they need to crossover to convert the hint to a basic feasible
    solution otherwise).

    Floating point values should be finite and not NaN, they are validated by
    MathOpt at Solve() time (resulting in an exception).

    Attributes:
      variable_values: a potentially partial assignment from the model's primal
        variables to finite (and not NaN) double values.
      dual_values: a potentially partial assignment from the model's linear
        constraints to finite (and not NaN) double values.
    """
    variable_values: Dict[model.Variable, float] = dataclasses.field(default_factory=dict)
    dual_values: Dict[model.LinearConstraint, float] = dataclasses.field(default_factory=dict)

    def to_proto(self) -> model_parameters_pb2.SolutionHintProto:
        """Returns an equivalent protocol buffer to this."""
        return model_parameters_pb2.SolutionHintProto(variable_values=sparse_containers.to_sparse_double_vector_proto(self.variable_values), dual_values=sparse_containers.to_sparse_double_vector_proto(self.dual_values))