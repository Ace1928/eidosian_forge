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
def _process_size(self, size):
    size = np.asarray(size)
    if size.ndim == 0:
        size = size[np.newaxis]
    elif size.ndim > 1:
        raise ValueError('Size must be an integer or tuple of integers; thus must have dimension <= 1. Got size.ndim = %s' % str(tuple(size)))
    n = size.prod()
    shape = tuple(size)
    return (n, shape)