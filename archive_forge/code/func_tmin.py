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
def tmin(a, lowerlimit=None, axis=0, inclusive=True):
    """
    Compute the trimmed minimum

    Parameters
    ----------
    a : array_like
        array of values
    lowerlimit : None or float, optional
        Values in the input array less than the given limit will be ignored.
        When lowerlimit is None, then all values are used. The default value
        is None.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    inclusive : {True, False}, optional
        This flag determines whether values exactly equal to the lower limit
        are included.  The default value is True.

    Returns
    -------
    tmin : float, int or ndarray

    Notes
    -----
    For more details on `tmin`, see `scipy.stats.tmin`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 2, 1, 2],
    ...               [8, 1, 8, 2],
    ...               [5, 3, 0, 2],
    ...               [4, 7, 5, 2]])
    ...
    >>> mstats.tmin(a, 5)
    masked_array(data=[5, 7, 5, --],
                 mask=[False, False, False,  True],
           fill_value=999999)

    """
    a, axis = _chk_asarray(a, axis)
    am = trima(a, (lowerlimit, None), (inclusive, False))
    return ma.minimum.reduce(am, axis)