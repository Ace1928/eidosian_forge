import warnings
from contextlib import nullcontext
from numpy.core import multiarray as mu
from numpy.core import umath as um
from numpy.core.multiarray import asanyarray
from numpy.core import numerictypes as nt
from numpy.core import _exceptions
from numpy.core._ufunc_config import _no_nep50_warning
from numpy._globals import _NoValue
from numpy.compat import pickle, os_fspath
def _count_reduce_items(arr, axis, keepdims=False, where=True):
    if where is True:
        if axis is None:
            axis = tuple(range(arr.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis,)
        items = 1
        for ax in axis:
            items *= arr.shape[mu.normalize_axis_index(ax, arr.ndim)]
        items = nt.intp(items)
    else:
        from numpy.lib.stride_tricks import broadcast_to
        items = umr_sum(broadcast_to(where, arr.shape), axis, nt.intp, None, keepdims)
    return items