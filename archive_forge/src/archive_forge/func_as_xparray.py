from __future__ import annotations
import os
import warnings
import numpy as np
from numpy.testing import assert_
import scipy._lib.array_api_compat.array_api_compat as array_api_compat
from scipy._lib.array_api_compat.array_api_compat import size
import scipy._lib.array_api_compat.array_api_compat.numpy as array_api_compat_numpy
def as_xparray(array, dtype=None, order=None, copy=None, *, xp=None, check_finite=False):
    """SciPy-specific replacement for `np.asarray` with `order` and `check_finite`.

    Memory layout parameter `order` is not exposed in the Array API standard.
    `order` is only enforced if the input array implementation
    is NumPy based, otherwise `order` is just silently ignored.

    `check_finite` is also not a keyword in the array API standard; included
    here for convenience rather than that having to be a separate function
    call inside SciPy functions.
    """
    if xp is None:
        xp = array_namespace(array)
    if xp.__name__ in {'numpy', 'scipy._lib.array_api_compat.array_api_compat.numpy'}:
        if copy is True:
            array = np.array(array, order=order, dtype=dtype)
        else:
            array = np.asarray(array, order=order, dtype=dtype)
        array = xp.asarray(array)
    else:
        try:
            array = xp.asarray(array, dtype=dtype, copy=copy)
        except TypeError:
            coerced_xp = array_namespace(xp.asarray(3))
            array = coerced_xp.asarray(array, dtype=dtype, copy=copy)
    if check_finite:
        _check_finite(array, xp)
    return array