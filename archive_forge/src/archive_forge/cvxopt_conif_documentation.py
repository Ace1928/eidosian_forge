from typing import Dict, List, Union
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, NonNeg, Zero
from cvxpy.reductions.solvers.compr_matrix import compress_matrix
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.kktsolver import setup_ldl_factor
Returns the KKT solver selected by the user.

        Removes the KKT solver from solver_opts.

        Parameters
        ----------
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        str or None
            The KKT solver chosen by the user.
        