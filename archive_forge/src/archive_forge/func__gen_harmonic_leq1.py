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
def _gen_harmonic_leq1(n, a):
    """Generalized harmonic number, a <= 1"""
    if not np.size(n):
        return n
    n_max = np.max(n)
    out = np.zeros_like(a, dtype=float)
    for i in np.arange(n_max, 0, -1, dtype=float):
        mask = i <= n
        out[mask] += 1 / i ** a[mask]
    return out