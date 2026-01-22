import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
def _nan_mask(a, out=None):
    """
    Parameters
    ----------
    a : array-like
        Input array with at least 1 dimension.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output and will prevent the allocation of a new array.

    Returns
    -------
    y : bool ndarray or True
        A bool array where ``np.nan`` positions are marked with ``False``
        and other positions are marked with ``True``. If the type of ``a``
        is such that it can't possibly contain ``np.nan``, returns ``True``.
    """
    if a.dtype.kind not in 'fc':
        return True
    y = np.isnan(a, out=out)
    y = np.invert(y, out=y)
    return y