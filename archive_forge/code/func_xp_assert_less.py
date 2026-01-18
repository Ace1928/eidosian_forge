from __future__ import annotations
import os
import warnings
import numpy as np
from numpy.testing import assert_
import scipy._lib.array_api_compat.array_api_compat as array_api_compat
from scipy._lib.array_api_compat.array_api_compat import size
import scipy._lib.array_api_compat.array_api_compat.numpy as array_api_compat_numpy
def xp_assert_less(actual, desired, check_namespace=True, check_dtype=True, check_shape=True, err_msg='', verbose=True, xp=None):
    if xp is None:
        xp = array_namespace(actual)
    desired = _strict_check(actual, desired, xp, check_namespace=check_namespace, check_dtype=check_dtype, check_shape=check_shape)
    if is_cupy(xp):
        return xp.testing.assert_array_less(actual, desired, err_msg=err_msg, verbose=verbose)
    elif is_torch(xp):
        if actual.device.type != 'cpu':
            actual = actual.cpu()
        if desired.device.type != 'cpu':
            desired = desired.cpu()
    return np.testing.assert_array_less(actual, desired, err_msg=err_msg, verbose=verbose)