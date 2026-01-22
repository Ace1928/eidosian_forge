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
@staticmethod
def SolveWithProto(model_request, response, interrupt=None):
    """
        Solves the model encoded by a MPModelRequest protocol buffer and fills the
        solution encoded as a MPSolutionResponse. The solve is stopped prematurely
        if interrupt is non-null at set to true during (or before) solving.
        Interruption is only supported if SolverTypeSupportsInterruption() returns
        true for the requested solver. Passing a non-null interruption with any
        other solver type immediately returns an MPSOLVER_INCOMPATIBLE_OPTIONS
        error.

        Note(user): This attempts to first use `DirectlySolveProto()` (if
        implemented). Consequently, this most likely does *not* override any of
        the default parameters of the underlying solver. This behavior *differs*
        from `MPSolver::Solve()` which by default sets the feasibility tolerance
        and the gap limit (as of 2020/02/11, to 1e-7 and 0.0001, respectively).
        """
    return _pywraplp.Solver_SolveWithProto(model_request, response, interrupt)