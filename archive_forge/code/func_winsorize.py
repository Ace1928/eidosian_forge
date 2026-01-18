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
def winsorize(a, limits=None, inclusive=(True, True), inplace=False, axis=None, nan_policy='propagate'):
    """Returns a Winsorized version of the input array.

    The (limits[0])th lowest values are set to the (limits[0])th percentile,
    and the (limits[1])th highest values are set to the (1 - limits[1])th
    percentile.
    Masked values are skipped.


    Parameters
    ----------
    a : sequence
        Input array.
    limits : {None, tuple of float}, optional
        Tuple of the percentages to cut on each side of the array, with respect
        to the number of unmasked data, as floats between 0. and 1.
        Noting n the number of unmasked data before trimming, the
        (n*limits[0])th smallest data and the (n*limits[1])th largest data are
        masked, and the total number of unmasked data after trimming
        is n*(1.-sum(limits)) The value of one limit can be set to None to
        indicate an open interval.
    inclusive : {(True, True) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be truncated (True) or rounded (False).
    inplace : {False, True}, optional
        Whether to winsorize in place (True) or to use a copy (False)
    axis : {None, int}, optional
        Axis along which to trim. If None, the whole array is trimmed, but its
        shape is maintained.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': allows nan values and may overwrite or propagate them
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Notes
    -----
    This function is applied to reduce the effect of possibly spurious outliers
    by limiting the extreme values.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import winsorize

    A shuffled array contains integers from 1 to 10.

    >>> a = np.array([10, 4, 9, 8, 5, 3, 7, 2, 1, 6])

    The 10% of the lowest value (i.e., `1`) and the 20% of the highest
    values (i.e., `9` and `10`) are replaced.

    >>> winsorize(a, limits=[0.1, 0.2])
    masked_array(data=[8, 4, 8, 8, 5, 3, 7, 2, 2, 6],
                 mask=False,
           fill_value=999999)

    """

    def _winsorize1D(a, low_limit, up_limit, low_include, up_include, contains_nan, nan_policy):
        n = a.count()
        idx = a.argsort()
        if contains_nan:
            nan_count = np.count_nonzero(np.isnan(a))
        if low_limit:
            if low_include:
                lowidx = int(low_limit * n)
            else:
                lowidx = np.round(low_limit * n).astype(int)
            if contains_nan and nan_policy == 'omit':
                lowidx = min(lowidx, n - nan_count - 1)
            a[idx[:lowidx]] = a[idx[lowidx]]
        if up_limit is not None:
            if up_include:
                upidx = n - int(n * up_limit)
            else:
                upidx = n - np.round(n * up_limit).astype(int)
            if contains_nan and nan_policy == 'omit':
                a[idx[upidx:-nan_count]] = a[idx[upidx - 1]]
            else:
                a[idx[upidx:]] = a[idx[upidx - 1]]
        return a
    contains_nan, nan_policy = _contains_nan(a, nan_policy)
    a = ma.array(a, copy=np.logical_not(inplace))
    if limits is None:
        return a
    if not isinstance(limits, tuple) and isinstance(limits, float):
        limits = (limits, limits)
    lolim, uplim = limits
    errmsg = 'The proportion to cut from the %s should be between 0. and 1.'
    if lolim is not None:
        if lolim > 1.0 or lolim < 0:
            raise ValueError(errmsg % 'beginning' + '(got %s)' % lolim)
    if uplim is not None:
        if uplim > 1.0 or uplim < 0:
            raise ValueError(errmsg % 'end' + '(got %s)' % uplim)
    loinc, upinc = inclusive
    if axis is None:
        shp = a.shape
        return _winsorize1D(a.ravel(), lolim, uplim, loinc, upinc, contains_nan, nan_policy).reshape(shp)
    else:
        return ma.apply_along_axis(_winsorize1D, axis, a, lolim, uplim, loinc, upinc, contains_nan, nan_policy)