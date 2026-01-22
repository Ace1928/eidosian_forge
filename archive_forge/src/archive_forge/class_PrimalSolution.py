import dataclasses
import enum
from typing import Dict, Optional, TypeVar
from ortools.math_opt import solution_pb2
from ortools.math_opt.python import model
from ortools.math_opt.python import sparse_containers
@dataclasses.dataclass
class PrimalSolution:
    """A solution to the optimization problem in a Model.

    E.g. consider a simple linear program:
      min c * x
      s.t. A * x >= b
      x >= 0.
    A primal solution is assignment values to x. It is feasible if it satisfies
    A * x >= b and x >= 0 from above. In the class PrimalSolution variable_values
    is x and objective_value is c * x.

    For the general case of a MathOpt optimization model, see go/mathopt-solutions
    for details.

    Attributes:
      variable_values: The value assigned for each Variable in the model.
      objective_value: The value of the objective value at this solution. This
        value may not be always populated.
      feasibility_status: The feasibility of the solution as claimed by the
        solver.
    """
    variable_values: Dict[model.Variable, float] = dataclasses.field(default_factory=dict)
    objective_value: float = 0.0
    feasibility_status: SolutionStatus = SolutionStatus.UNDETERMINED

    def to_proto(self) -> solution_pb2.PrimalSolutionProto:
        """Returns an equivalent proto for a primal solution."""
        return solution_pb2.PrimalSolutionProto(variable_values=sparse_containers.to_sparse_double_vector_proto(self.variable_values), objective_value=self.objective_value, feasibility_status=self.feasibility_status.value)