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
def SetHint(self, variables, values):
    """
        Set a hint for solution.

        If a feasible or almost-feasible solution to the problem is already known,
        it may be helpful to pass it to the solver so that it can be used. A
        solver that supports this feature will try to use this information to
        create its initial feasible solution.

        Note that it may not always be faster to give a hint like this to the
        solver. There is also no guarantee that the solver will use this hint or
        try to return a solution "close" to this assignment in case of multiple
        optimal solutions.
        """
    return _pywraplp.Solver_SetHint(self, variables, values)