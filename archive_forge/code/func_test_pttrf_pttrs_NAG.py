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
@pytest.mark.parametrize('d, e, d_expect, e_expect, b, x_expect', [(np.array([4, 10, 29, 25, 5]), np.array([-2, -6, 15, 8]), np.array([4, 9, 25, 16, 1]), np.array([-0.5, -0.6667, 0.6, 0.5]), np.array([[6, 10], [9, 4], [2, 9], [14, 65], [7, 23]]), np.array([[2.5, 2], [2, -1], [1, -3], [-1, 6], [3, -5]])), (np.array([16, 41, 46, 21]), np.array([16 + 16j, 18 - 9j, 1 - 4j]), np.array([16, 9, 1, 4]), np.array([1 + 1j, 2 - 1j, 1 - 4j]), np.array([[64 + 16j, -16 - 32j], [93 + 62j, 61 - 66j], [78 - 80j, 71 - 74j], [14 - 27j, 35 + 15j]]), np.array([[2 + 1j, -3 - 2j], [1 + 1j, 1 + 1j], [1 - 2j, 1 - 2j], [1 - 1j, 2 + 1j]]))])
def test_pttrf_pttrs_NAG(d, e, d_expect, e_expect, b, x_expect):
    atol = 0.0001
    pttrf = get_lapack_funcs('pttrf', dtype=e[0])
    _d, _e, info = pttrf(d, e)
    assert_allclose(_d, d_expect, atol=atol)
    assert_allclose(_e, e_expect, atol=atol)
    pttrs = get_lapack_funcs('pttrs', dtype=e[0])
    _x, info = pttrs(_d, _e.conj(), b)
    assert_allclose(_x, x_expect, atol=atol)
    if e.dtype in COMPLEX_DTYPES:
        _x, info = pttrs(_d, _e, b, lower=1)
        assert_allclose(_x, x_expect, atol=atol)