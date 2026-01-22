import warnings
import numpy as np
import scipy  # For version checks
import cvxpy.settings as s
from cvxpy.constraints import NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version
def _log_scipy_method_warning(self, meth):
    warnings.warn("It is best to specify the 'method' parameter within scipy_options. The main advantage of this solver is its ability to use the HiGHS LP solvers via scipy.optimize.linprog(), which requires a SciPy version >= 1.6.1.\n\nThe default method '{}' will be used in this case.\n".format(meth))