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
def tmean(a, limits=None, inclusive=(True, True), axis=None):
    """
    Compute the trimmed mean.

    Parameters
    ----------
    a : array_like
        Array of values.
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored.  When limits is None (default), then all
        values are used.  Either of the limit values in the tuple can also be
        None representing a half-open interval.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. If None, compute over the
        whole array. Default is None.

    Returns
    -------
    tmean : float

    Notes
    -----
    For more details on `tmean`, see `scipy.stats.tmean`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 9, 1, 2],
    ...               [8, 7, 8, 2],
    ...               [5, 6, 0, 2],
    ...               [4, 5, 5, 2]])
    ...
    ...
    >>> mstats.tmean(a, (2,5))
    3.3
    >>> mstats.tmean(a, (2,5), axis=0)
    masked_array(data=[4.0, 5.0, 4.0, 2.0],
                 mask=[False, False, False, False],
           fill_value=1e+20)

    """
    return trima(a, limits=limits, inclusive=inclusive).mean(axis=axis)