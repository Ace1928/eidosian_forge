import numpy as np
import scipy as sp
import scipy.sparse as sps
from warnings import warn
from scipy.linalg import LinAlgError
from ._optimize import OptimizeWarning, OptimizeResult, _check_unknown_options
from ._linprog_util import _postsolve
def _sym_solve(Dinv, A, r1, r2, solve):
    """
    An implementation of [4] equation 8.31 and 8.32

    References
    ----------
    .. [4] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    r = r2 + A.dot(Dinv * r1)
    v = solve(r)
    u = Dinv * (A.T.dot(v) - r1)
    return (u, v)