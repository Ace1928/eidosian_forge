import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def _hdsd_1D(data, prob):
    """Computes the std error for 1D arrays."""
    xsorted = np.sort(data.compressed())
    n = len(xsorted)
    hdsd = np.empty(len(prob), float64)
    if n < 2:
        hdsd.flat = np.nan
    vv = np.arange(n) / float(n - 1)
    betacdf = beta.cdf
    for i, p in enumerate(prob):
        _w = betacdf(vv, n * p, n * (1 - p))
        w = _w[1:] - _w[:-1]
        mx_ = np.zeros_like(xsorted)
        mx_[1:] = np.cumsum(w * xsorted[:-1])
        mx_[:-1] += np.cumsum(w[::-1] * xsorted[:0:-1])[::-1]
        hdsd[i] = np.sqrt(mx_.var() * (n - 1))
    return hdsd