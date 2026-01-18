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
def tmax(a, upperlimit=None, axis=0, inclusive=True):
    """
    Compute the trimmed maximum

    This function computes the maximum value of an array along a given axis,
    while ignoring values larger than a specified upper limit.

    Parameters
    ----------
    a : array_like
        array of values
    upperlimit : None or float, optional
        Values in the input array greater than the given limit will be ignored.
        When upperlimit is None, then all values are used. The default value
        is None.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    inclusive : {True, False}, optional
        This flag determines whether values exactly equal to the upper limit
        are included.  The default value is True.

    Returns
    -------
    tmax : float, int or ndarray

    Notes
    -----
    For more details on `tmax`, see `scipy.stats.tmax`.

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
    >>> mstats.tmax(a, 4)
    masked_array(data=[4, --, 3, 2],
                 mask=[False,  True, False, False],
           fill_value=999999)

    """
    a, axis = _chk_asarray(a, axis)
    am = trima(a, (None, upperlimit), (False, inclusive))
    return ma.maximum.reduce(am, axis)