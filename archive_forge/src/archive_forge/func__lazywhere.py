import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
import numpy as np
from scipy._lib._array_api import array_namespace
def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
    """Return elements chosen from two possibilities depending on a condition

    Equivalent to ``f(*arrays) if cond else fillvalue`` performed elementwise.

    Parameters
    ----------
    cond : array
        The condition (expressed as a boolean array).
    arrays : tuple of array
        Arguments to `f` (and `f2`). Must be broadcastable with `cond`.
    f : callable
        Where `cond` is True, output will be ``f(arr1[cond], arr2[cond], ...)``
    fillvalue : object
        If provided, value with which to fill output array where `cond` is
        not True.
    f2 : callable
        If provided, output will be ``f2(arr1[cond], arr2[cond], ...)`` where
        `cond` is not True.

    Returns
    -------
    out : array
        An array with elements from the output of `f` where `cond` is True
        and `fillvalue` (or elements from the output of `f2`) elsewhere. The
        returned array has data type determined by Type Promotion Rules
        with the output of `f` and `fillvalue` (or the output of `f2`).

    Notes
    -----
    ``xp.where(cond, x, fillvalue)`` requires explicitly forming `x` even where
    `cond` is False. This function evaluates ``f(arr1[cond], arr2[cond], ...)``
    onle where `cond` ``is True.

    Examples
    --------
    >>> import numpy as np
    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
    >>> def f(a, b):
    ...     return a*b
    >>> _lazywhere(a > 2, (a, b), f, np.nan)
    array([ nan,  nan,  21.,  32.])

    """
    xp = array_namespace(cond, *arrays)
    if f2 is fillvalue is None or (f2 is not None and fillvalue is not None):
        raise ValueError('Exactly one of `fillvalue` or `f2` must be given.')
    args = xp.broadcast_arrays(cond, *arrays)
    cond, arrays = (xp.astype(args[0], bool, copy=False), args[1:])
    temp1 = xp.asarray(f(*(arr[cond] for arr in arrays)))
    if f2 is None:
        fillvalue = xp.asarray(fillvalue)
        dtype = xp.result_type(temp1.dtype, fillvalue.dtype)
        out = xp.full(cond.shape, fill_value=fillvalue, dtype=dtype)
    else:
        ncond = ~cond
        temp2 = xp.asarray(f2(*(arr[ncond] for arr in arrays)))
        dtype = xp.result_type(temp1, temp2)
        out = xp.empty(cond.shape, dtype=dtype)
        out[ncond] = temp2
    out[cond] = temp1
    return out