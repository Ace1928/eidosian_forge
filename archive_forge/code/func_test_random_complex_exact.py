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
@pytest.mark.skipif(IS_MUSL, reason='may segfault on Alpine, see gh-17630')
def test_random_complex_exact(self):
    rng = np.random.RandomState(1234)
    for dtype in COMPLEX_DTYPES:
        for n in (20, 200):
            for lapack_driver in TestLstsq.lapack_drivers:
                for overwrite in (True, False):
                    a = np.asarray(rng.random([n, n]) + 1j * rng.random([n, n]), dtype=dtype)
                    for i in range(n):
                        a[i, i] = 20 * (0.1 + a[i, i])
                    for i in range(2):
                        b = np.asarray(rng.random([n, 3]), dtype=dtype)
                        a1 = a.copy()
                        b1 = b.copy()
                        out = lstsq(a1, b1, lapack_driver=lapack_driver, overwrite_a=overwrite, overwrite_b=overwrite)
                        x = out[0]
                        r = out[2]
                        assert_(r == n, f'expected efficient rank {n}, got {r}')
                        if dtype is np.complex64:
                            assert_allclose(dot(a, x), b, rtol=400 * _eps_cast(a1.dtype), atol=400 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)
                        else:
                            assert_allclose(dot(a, x), b, rtol=1000 * _eps_cast(a1.dtype), atol=1000 * _eps_cast(a1.dtype), err_msg='driver: %s' % lapack_driver)