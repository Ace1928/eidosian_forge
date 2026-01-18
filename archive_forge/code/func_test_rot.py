import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
def test_rot():
    for dtype in 'fdFD':
        c = 0.6
        s = 0.8
        u = np.full(4, 3, dtype)
        v = np.full(4, 4, dtype)
        atol = 10 ** (-(np.finfo(dtype).precision - 1))
        if dtype in 'fd':
            rot = get_blas_funcs('rot', dtype=dtype)
            f = 4
        else:
            rot = get_lapack_funcs('rot', dtype=dtype)
            s *= -1j
            v *= 1j
            f = 4j
        assert_allclose(rot(u, v, c, s), [[5, 5, 5, 5], [0, 0, 0, 0]], atol=atol)
        assert_allclose(rot(u, v, c, s, n=2), [[5, 5, 3, 3], [0, 0, f, f]], atol=atol)
        assert_allclose(rot(u, v, c, s, offx=2, offy=2), [[3, 3, 5, 5], [f, f, 0, 0]], atol=atol)
        assert_allclose(rot(u, v, c, s, incx=2, offy=2, n=2), [[5, 3, 5, 3], [f, f, 0, 0]], atol=atol)
        assert_allclose(rot(u, v, c, s, offx=2, incy=2, n=2), [[3, 3, 5, 5], [0, f, 0, f]], atol=atol)
        assert_allclose(rot(u, v, c, s, offx=2, incx=2, offy=2, incy=2, n=1), [[3, 3, 5, 3], [f, f, 0, f]], atol=atol)
        assert_allclose(rot(u, v, c, s, incx=-2, incy=-2, n=2), [[5, 3, 5, 3], [0, f, 0, f]], atol=atol)
        a, b = rot(u, v, c, s, overwrite_x=1, overwrite_y=1)
        assert_(a is u)
        assert_(b is v)
        assert_allclose(a, [5, 5, 5, 5], atol=atol)
        assert_allclose(b, [0, 0, 0, 0], atol=atol)