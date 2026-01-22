import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def _interval_contains_close(interval, val, rtol=1e-10):
    """
    Check, inclusively, whether an interval includes a given value, with the
    interval expanded by a small tolerance to admit floating point errors.

    Parameters
    ----------
    interval : (float, float)
        The endpoints of the interval.
    val : float
        Value to check is within interval.
    rtol : float, default: 1e-10
        Relative tolerance slippage allowed outside of the interval.
        For an interval ``[a, b]``, values
        ``a - rtol * (b - a) <= val <= b + rtol * (b - a)`` are considered
        inside the interval.

    Returns
    -------
    bool
        Whether *val* is within the *interval* (with tolerance).
    """
    a, b = interval
    if a > b:
        a, b = (b, a)
    rtol = (b - a) * rtol
    return a - rtol <= val <= b + rtol