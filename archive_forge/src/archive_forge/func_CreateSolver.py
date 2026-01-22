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
def CreateSolver(solver_id):
    """
        Recommended factory method to create a MPSolver instance, especially in
        non C++ languages.

        It returns a newly created solver instance if successful, or a nullptr
        otherwise. This can occur if the relevant interface is not linked in, or if
        a needed license is not accessible for commercial solvers.

        Ownership of the solver is passed on to the caller of this method.
        It will accept both string names of the OptimizationProblemType enum, as
        well as a short version (i.e. "SCIP_MIXED_INTEGER_PROGRAMMING" or "SCIP").

        solver_id is case insensitive, and the following names are supported:
          - CLP_LINEAR_PROGRAMMING or CLP
          - CBC_MIXED_INTEGER_PROGRAMMING or CBC
          - GLOP_LINEAR_PROGRAMMING or GLOP
          - BOP_INTEGER_PROGRAMMING or BOP
          - SAT_INTEGER_PROGRAMMING or SAT or CP_SAT
          - SCIP_MIXED_INTEGER_PROGRAMMING or SCIP
          - GUROBI_LINEAR_PROGRAMMING or GUROBI_LP
          - GUROBI_MIXED_INTEGER_PROGRAMMING or GUROBI or GUROBI_MIP
          - CPLEX_LINEAR_PROGRAMMING or CPLEX_LP
          - CPLEX_MIXED_INTEGER_PROGRAMMING or CPLEX or CPLEX_MIP
          - XPRESS_LINEAR_PROGRAMMING or XPRESS_LP
          - XPRESS_MIXED_INTEGER_PROGRAMMING or XPRESS or XPRESS_MIP
          - GLPK_LINEAR_PROGRAMMING or GLPK_LP
          - GLPK_MIXED_INTEGER_PROGRAMMING or GLPK or GLPK_MIP
        """
    return _pywraplp.Solver_CreateSolver(solver_id)