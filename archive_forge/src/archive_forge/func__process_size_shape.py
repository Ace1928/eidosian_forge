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
@staticmethod
def _process_size_shape(size, r, c):
    """
        Compute the number of samples to be drawn and the shape of the output
        """
    shape = (len(r), len(c))
    if size is None:
        return (1, shape)
    size = np.atleast_1d(size)
    if not np.issubdtype(size.dtype, np.integer) or np.any(size < 0):
        raise ValueError('`size` must be a non-negative integer or `None`')
    return (np.prod(size), tuple(size) + shape)