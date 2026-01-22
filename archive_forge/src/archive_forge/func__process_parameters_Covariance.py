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
def _process_parameters_Covariance(self, mean, cov):
    dim = cov.shape[-1]
    mean = np.array([0.0]) if mean is None else mean
    message = f'`cov` represents a covariance matrix in {dim} dimensions,and so `mean` must be broadcastable to shape {(dim,)}'
    try:
        mean = np.broadcast_to(mean, dim)
    except ValueError as e:
        raise ValueError(message) from e
    return (dim, mean, cov)