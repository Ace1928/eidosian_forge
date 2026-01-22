import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
def _nanquantile_ureduce_func(a, q, axis=None, out=None, overwrite_input=False, method='linear'):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    """
    if axis is None or a.ndim == 1:
        part = a.ravel()
        result = _nanquantile_1d(part, q, overwrite_input, method)
    else:
        result = np.apply_along_axis(_nanquantile_1d, axis, a, q, overwrite_input, method)
        if q.ndim != 0:
            result = np.moveaxis(result, axis, 0)
    if out is not None:
        out[...] = result
    return result