import warnings
from . import _minpack
import numpy as np
from numpy import (atleast_1d, triu, shape, transpose, zeros, prod, greater,
from scipy import linalg
from scipy.linalg import svd, cholesky, solve_triangular, LinAlgError
from scipy._lib._util import _asarray_validated, _lazywhere, _contains_nan
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._optimize import OptimizeResult, _check_unknown_options, OptimizeWarning
from ._lsq import least_squares
from ._lsq.least_squares import prepare_bounds
from scipy.optimize._minimize import Bounds
from numpy import dot, eye, take  # noqa: F401
from numpy.linalg import inv  # noqa: F401
def _initialize_feasible(lb, ub):
    p0 = np.ones_like(lb)
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)
    mask = lb_finite & ub_finite
    p0[mask] = 0.5 * (lb[mask] + ub[mask])
    mask = lb_finite & ~ub_finite
    p0[mask] = lb[mask] + 1
    mask = ~lb_finite & ub_finite
    p0[mask] = ub[mask] - 1
    return p0