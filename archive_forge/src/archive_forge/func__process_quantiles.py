import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
def _process_quantiles(self, x, M, m, n):
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.integer):
        raise TypeError("'x' must an array of integers.")
    if x.ndim == 0:
        raise ValueError("'x' must be an array with at least one dimension.")
    if not x.shape[-1] == m.shape[-1]:
        raise ValueError(f"Size of each quantile must be size of 'm': received {x.shape[-1]}, but expected {m.shape[-1]}.")
    if m.size != 0:
        n = n[..., np.newaxis]
        M = M[..., np.newaxis]
    x, m, n, M = np.broadcast_arrays(x, m, n, M)
    if m.size != 0:
        n, M = (n[..., 0], M[..., 0])
    xcond = (x < 0) | (x > m)
    return (x, M, m, n, xcond, np.any(xcond, axis=-1) | (x.sum(axis=-1) != n))