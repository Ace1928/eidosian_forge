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
@staticmethod
def _restore_solver_options(old_options) -> None:
    import cvxopt.solvers
    for key, value in list(cvxopt.solvers.options.items()):
        if key in old_options:
            cvxopt.solvers.options[key] = old_options[key]
        else:
            del cvxopt.solvers.options[key]