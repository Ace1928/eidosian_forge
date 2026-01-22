from functools import partial
from scipy import special
from scipy.special import entr, logsumexp, betaln, gammaln as gamln, zeta
from scipy._lib._util import _lazywhere, rng_integers
from scipy.interpolate import interp1d
from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh
import numpy as np
from ._distn_infrastructure import (rv_discrete, get_distribution_names,
import scipy.stats._boost as _boost
from ._biasedurn import (_PyFishersNCHypergeometric,
def _vectorize_rvs_over_shapes(_rvs1):
    """Decorator that vectorizes _rvs method to work on ndarray shapes"""

    def _rvs(*args, size, random_state):
        _rvs1_size, _rvs1_indices = _check_shape(args[0].shape, size)
        size = np.array(size)
        _rvs1_size = np.array(_rvs1_size)
        _rvs1_indices = np.array(_rvs1_indices)
        if np.all(_rvs1_indices):
            return _rvs1(*args, size, random_state)
        out = np.empty(size)
        j0 = np.arange(out.ndim)
        j1 = np.hstack((j0[~_rvs1_indices], j0[_rvs1_indices]))
        out = np.moveaxis(out, j1, j0)
        for i in np.ndindex(*size[~_rvs1_indices]):
            out[i] = _rvs1(*[np.squeeze(arg)[i] for arg in args], _rvs1_size, random_state)
        return np.moveaxis(out, j0, j1)
    return _rvs