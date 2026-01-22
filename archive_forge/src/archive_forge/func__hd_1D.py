import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def _hd_1D(data, prob, var):
    """Computes the HD quantiles for a 1D array. Returns nan for invalid data."""
    xsorted = np.squeeze(np.sort(data.compressed().view(ndarray)))
    n = xsorted.size
    hd = np.empty((2, len(prob)), float64)
    if n < 2:
        hd.flat = np.nan
        if var:
            return hd
        return hd[0]
    v = np.arange(n + 1) / float(n)
    betacdf = beta.cdf
    for i, p in enumerate(prob):
        _w = betacdf(v, (n + 1) * p, (n + 1) * (1 - p))
        w = _w[1:] - _w[:-1]
        hd_mean = np.dot(w, xsorted)
        hd[0, i] = hd_mean
        hd[1, i] = np.dot(w, (xsorted - hd_mean) ** 2)
    hd[0, prob == 0] = xsorted[0]
    hd[0, prob == 1] = xsorted[-1]
    if var:
        hd[1, prob == 0] = hd[1, prob == 1] = np.nan
        return hd
    return hd[0]