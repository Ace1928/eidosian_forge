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
def _gen_harmonic(n, a):
    """Generalized harmonic number"""
    n, a = np.broadcast_arrays(n, a)
    return _lazywhere(a > 1, (n, a), f=_gen_harmonic_gt1, f2=_gen_harmonic_leq1)