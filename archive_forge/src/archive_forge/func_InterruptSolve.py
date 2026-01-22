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
def InterruptSolve(self):
    """
         Interrupts the Solve() execution to terminate processing if possible.

        If the underlying interface supports interruption; it does that and returns
        true regardless of whether there's an ongoing Solve() or not. The Solve()
        call may still linger for a while depending on the conditions.  If
        interruption is not supported; returns false and does nothing.
        MPSolver::SolverTypeSupportsInterruption can be used to check if
        interruption is supported for a given solver type.
        """
    return _pywraplp.Solver_InterruptSolve(self)