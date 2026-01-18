import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def tsem(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    """
    Compute the trimmed standard error of the mean.

    This function finds the standard error of the mean for given
    values, ignoring values outside the given `limits`.

    Parameters
    ----------
    a : array_like
        array of values
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored. When limits is None, then all values are
        used. Either of the limit values in the tuple can also be None
        representing a half-open interval.  The default value is None.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. If None, compute over the
        whole array. Default is zero.
    ddof : int, optional
        Delta degrees of freedom. Default is 1.

    Returns
    -------
    tsem : float

    Notes
    -----
    For more details on `tsem`, see `scipy.stats.tsem`.

    """
    a = ma.asarray(a).ravel()
    if limits is None:
        n = float(a.count())
        return a.std(axis=axis, ddof=ddof) / ma.sqrt(n)
    am = trima(a.ravel(), limits, inclusive)
    sd = np.sqrt(am.var(axis=axis, ddof=ddof))
    return sd / np.sqrt(am.count())