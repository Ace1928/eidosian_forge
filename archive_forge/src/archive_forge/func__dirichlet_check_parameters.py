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
def _dirichlet_check_parameters(alpha):
    alpha = np.asarray(alpha)
    if np.min(alpha) <= 0:
        raise ValueError('All parameters must be greater than 0')
    elif alpha.ndim != 1:
        raise ValueError(f"Parameter vector 'a' must be one dimensional, but a.shape = {alpha.shape}.")
    return alpha