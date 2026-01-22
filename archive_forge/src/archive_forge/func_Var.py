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
def Var(self, lb, ub, integer, name):
    """
        Creates a variable with the given bounds, integrality requirement and
        name. Bounds can be finite or +/- MPSolver::infinity(). The MPSolver owns
        the variable (i.e. the returned pointer is borrowed). Variable names are
        optional. If you give an empty name, name() will auto-generate one for you
        upon request.
        """
    return _pywraplp.Solver_Var(self, lb, ub, integer, name)