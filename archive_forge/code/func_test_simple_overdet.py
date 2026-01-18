import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
def test_simple_overdet(self):
    for dtype in REAL_DTYPES:
        a = np.array([[1, 2], [4, 5], [3, 4]], dtype=dtype)
        b = np.array([1, 2, 3], dtype=dtype)
        for lapack_driver in TestLstsq.lapack_drivers:
            for overwrite in (True, False):
                a1 = a.copy()
                b1 = b.copy()
                out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                x = out[0]
                if lapack_driver == 'gelsy':
                    residuals = np.sum((b - a.dot(x)) ** 2)
                else:
                    residuals = out[1]
                r = out[2]
                assert_(r == 2, 'expected efficient rank 2, got %s' % r)
                assert_allclose(abs((dot(a, x) - b) ** 2).sum(axis=0), residuals, rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)
                assert_allclose(x, (-0.428571428571429, 0.85714285714285), rtol=25 * _eps_cast(a1.dtype), atol=25 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)