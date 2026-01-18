from __future__ import annotations
import os
import warnings
import numpy as np
from numpy.testing import assert_
import scipy._lib.array_api_compat.array_api_compat as array_api_compat
from scipy._lib.array_api_compat.array_api_compat import size
import scipy._lib.array_api_compat.array_api_compat.numpy as array_api_compat_numpy
def xp_assert_close(actual, desired, rtol=1e-07, atol=0, check_namespace=True, check_dtype=True, check_shape=True, err_msg='', xp=None):
    if xp is None:
        xp = array_namespace(actual)
    desired = _strict_check(actual, desired, xp, check_namespace=check_namespace, check_dtype=check_dtype, check_shape=check_shape)
    if is_cupy(xp):
        return xp.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=err_msg)
    elif is_torch(xp):
        err_msg = None if err_msg == '' else err_msg
        return xp.testing.assert_close(actual, desired, rtol=rtol, atol=atol, equal_nan=True, check_dtype=False, msg=err_msg)
    return np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=err_msg)