import numpy as np
from numpy import float64, ndarray
import numpy.ma as ma
from numpy.ma import MaskedArray
from . import _mstats_basic as mstats
from scipy.stats.distributions import norm, beta, t, binom
def trimmed_mean_ci(data, limits=(0.2, 0.2), inclusive=(True, True), alpha=0.05, axis=None):
    """
    Selected confidence interval of the trimmed mean along the given axis.

    Parameters
    ----------
    data : array_like
        Input data.
    limits : {None, tuple}, optional
        None or a two item tuple.
        Tuple of the percentages to cut on each side of the array, with respect
        to the number of unmasked data, as floats between 0. and 1. If ``n``
        is the number of unmasked data before trimming, then
        (``n * limits[0]``)th smallest data and (``n * limits[1]``)th
        largest data are masked.  The total number of unmasked data after
        trimming is ``n * (1. - sum(limits))``.
        The value of one limit can be set to None to indicate an open interval.

        Defaults to (0.2, 0.2).
    inclusive : (2,) tuple of boolean, optional
        If relative==False, tuple indicating whether values exactly equal to
        the absolute limits are allowed.
        If relative==True, tuple indicating whether the number of data being
        masked on each side should be rounded (True) or truncated (False).

        Defaults to (True, True).
    alpha : float, optional
        Confidence level of the intervals.

        Defaults to 0.05.
    axis : int, optional
        Axis along which to cut. If None, uses a flattened version of `data`.

        Defaults to None.

    Returns
    -------
    trimmed_mean_ci : (2,) ndarray
        The lower and upper confidence intervals of the trimmed data.

    """
    data = ma.array(data, copy=False)
    trimmed = mstats.trimr(data, limits=limits, inclusive=inclusive, axis=axis)
    tmean = trimmed.mean(axis)
    tstde = mstats.trimmed_stde(data, limits=limits, inclusive=inclusive, axis=axis)
    df = trimmed.count(axis) - 1
    tppf = t.ppf(1 - alpha / 2.0, df)
    return np.array((tmean - tppf * tstde, tmean + tppf * tstde))