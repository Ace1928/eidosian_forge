from sys import version_info as _swig_python_version_info
import numbers
from ortools.linear_solver.python.linear_solver_natural_api import OFFSET_KEY
from ortools.linear_solver.python.linear_solver_natural_api import inf
from ortools.linear_solver.python.linear_solver_natural_api import LinearExpr
from ortools.linear_solver.python.linear_solver_natural_api import ProductCst
from ortools.linear_solver.python.linear_solver_natural_api import Sum
from ortools.linear_solver.python.linear_solver_natural_api import SumArray
from ortools.linear_solver.python.linear_solver_natural_api import SumCst
from ortools.linear_solver.python.linear_solver_natural_api import LinearConstraint
from ortools.linear_solver.python.linear_solver_natural_api import VariableExpr
def VerifySolution(self, tolerance, log_errors):
    """
        Advanced usage: Verifies the *correctness* of the solution.

        It verifies that all variables must be within their domains, all
        constraints must be satisfied, and the reported objective value must be
        accurate.

        Usage:
        - This can only be called after Solve() was called.
        - "tolerance" is interpreted as an absolute error threshold.
        - For the objective value only, if the absolute error is too large,
          the tolerance is interpreted as a relative error threshold instead.
        - If "log_errors" is true, every single violation will be logged.
        - If "tolerance" is negative, it will be set to infinity().

        Most users should just set the --verify_solution flag and not bother using
        this method directly.
        """
    return _pywraplp.Solver_VerifySolution(self, tolerance, log_errors)