import collections.abc
import functools
import re
import sys
import warnings
from .._utils import set_module
import numpy as np
import numpy.core.numeric as _nx
from numpy.core import transpose
from numpy.core.numeric import (
from numpy.core.umath import (
from numpy.core.fromnumeric import (
from numpy.core.numerictypes import typecodes
from numpy.core import overrides
from numpy.core.function_base import add_newdoc
from numpy.lib.twodim_base import diag
from numpy.core.multiarray import (
from numpy.core.umath import _add_newdoc_ufunc as add_newdoc_ufunc
import builtins
from numpy.lib.histograms import histogram, histogramdd  # noqa: F401
def _ureduce(a, func, keepdims=False, **kwargs):
    """
    Internal Function.
    Call `func` with `a` as first argument swapping the axes to use extended
    axis on functions that don't support it natively.

    Returns result and a.shape with axis dims set to 1.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    func : callable
        Reduction function capable of receiving a single axis argument.
        It is called with `a` as first argument followed by `kwargs`.
    kwargs : keyword arguments
        additional keyword arguments to pass to `func`.

    Returns
    -------
    result : tuple
        Result of func(a, **kwargs) and a.shape with axis dims set to 1
        which can be used to reshape the result to the same shape a ufunc with
        keepdims=True would produce.

    """
    a = np.asanyarray(a)
    axis = kwargs.get('axis', None)
    out = kwargs.get('out', None)
    if keepdims is np._NoValue:
        keepdims = False
    nd = a.ndim
    if axis is not None:
        axis = _nx.normalize_axis_tuple(axis, nd)
        if keepdims:
            if out is not None:
                index_out = tuple((0 if i in axis else slice(None) for i in range(nd)))
                kwargs['out'] = out[(Ellipsis,) + index_out]
        if len(axis) == 1:
            kwargs['axis'] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs['axis'] = -1
    elif keepdims:
        if out is not None:
            index_out = (0,) * nd
            kwargs['out'] = out[(Ellipsis,) + index_out]
    r = func(a, **kwargs)
    if out is not None:
        return out
    if keepdims:
        if axis is None:
            index_r = (np.newaxis,) * nd
        else:
            index_r = tuple((np.newaxis if i in axis else slice(None) for i in range(nd)))
        r = r[(Ellipsis,) + index_r]
    return r