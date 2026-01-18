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
def trimboth(data, proportiontocut=0.2, inclusive=(True, True), axis=None):
    """
    Trims the smallest and largest data values.

    Trims the `data` by masking the ``int(proportiontocut * n)`` smallest and
    ``int(proportiontocut * n)`` largest values of data along the given axis,
    where n is the number of unmasked values before trimming.

    Parameters
    ----------
    data : ndarray
        Data to trim.
    proportiontocut : float, optional
        Percentage of trimming (as a float between 0 and 1).
        If n is the number of unmasked values before trimming, the number of
        values after trimming is ``(1 - 2*proportiontocut) * n``.
        Default is 0.2.
    inclusive : {(bool, bool) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).
    axis : int, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.

    """
    return trimr(data, limits=(proportiontocut, proportiontocut), inclusive=inclusive, axis=axis)