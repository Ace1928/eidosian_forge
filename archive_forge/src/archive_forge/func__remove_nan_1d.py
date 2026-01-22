import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
def _remove_nan_1d(arr1d, overwrite_input=False):
    """
    Equivalent to arr1d[~arr1d.isnan()], but in a different order

    Presumably faster as it incurs fewer copies

    Parameters
    ----------
    arr1d : ndarray
        Array to remove nans from
    overwrite_input : bool
        True if `arr1d` can be modified in place

    Returns
    -------
    res : ndarray
        Array with nan elements removed
    overwrite_input : bool
        True if `res` can be modified in place, given the constraint on the
        input
    """
    if arr1d.dtype == object:
        c = np.not_equal(arr1d, arr1d, dtype=bool)
    else:
        c = np.isnan(arr1d)
    s = np.nonzero(c)[0]
    if s.size == arr1d.size:
        warnings.warn('All-NaN slice encountered', RuntimeWarning, stacklevel=6)
        return (arr1d[:0], True)
    elif s.size == 0:
        return (arr1d, overwrite_input)
    else:
        if not overwrite_input:
            arr1d = arr1d.copy()
        enonan = arr1d[-s.size:][~c[-s.size:]]
        arr1d[s[:enonan.size]] = enonan
        return (arr1d[:-s.size], True)