import warnings
import cvxpy.error as error
import cvxpy.problems as problems
import cvxpy.settings as s
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.solution import failure_solution
def _lower_problem(problem):
    """Evaluates lazy constraints."""
    return problems.problem.Problem(Minimize(0), problem.constraints + [c() for c in problem._lazy_constraints])