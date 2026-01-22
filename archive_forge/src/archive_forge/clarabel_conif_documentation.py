import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.expressions.expression import Expression
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.versioning import Version
Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : Bool
            Whether to warm_start Clarabel.
            PJG: From SCS.   We don't support this, not sure if relevant
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            Clarabel-specific solver options.

        Returns
        -------
        The result returned by a call to clarabel.solve().
        