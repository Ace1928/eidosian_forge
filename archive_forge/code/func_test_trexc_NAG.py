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
@pytest.mark.parametrize('t, expect, ifst, ilst', [(np.array([[0.8, -0.11, 0.01, 0.03], [0.0, -0.1, 0.25, 0.35], [0.0, -0.65, -0.1, 0.2], [0.0, 0.0, 0.0, -0.1]]), np.array([[-0.1, -0.6463, 0.0874, 0.201], [0.2514, -0.1, 0.0927, 0.3505], [0.0, 0.0, 0.8, -0.0117], [0.0, 0.0, 0.0, -0.1]]), 2, 1), (np.array([[-6.0 - 7j, 0.36 - 0.36j, -0.19 + 0.48j, 0.88 - 0.25j], [0.0 + 0j, -5.0 + 2j, -0.03 - 0.72j, -0.23 + 0.13j], [0.0 + 0j, 0.0 + 0j, 8.0 - 1j, 0.94 + 0.53j], [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 3.0 - 4j]]), np.array([[-5.0 + 2j, -0.1574 + 0.7143j, 0.1781 - 0.1913j, 0.395 + 0.3861j], [0.0 + 0j, 8.0 - 1j, 1.0742 + 0.1447j, 0.2515 - 0.3397j], [0.0 + 0j, 0.0 + 0j, 3.0 - 4j, 0.2264 + 0.8962j], [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, -6.0 - 7j]]), 1, 4)])
def test_trexc_NAG(t, ifst, ilst, expect):
    """
    This test implements the example found in the NAG manual,
    f08qfc, f08qtc, f08qgc, f08quc.
    """
    atol = 0.0001
    trexc = get_lapack_funcs('trexc', dtype=t.dtype)
    result = trexc(t, t, ifst, ilst, wantq=0)
    assert_equal(result[-1], 0)
    t = result[0]
    assert_allclose(expect, t, atol=atol)