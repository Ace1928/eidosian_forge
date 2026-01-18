import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2
@pytest.mark.slow
def test_onenormest_linear_operator(self):
    np.random.seed(1234)
    n = 6000
    k = 3
    A = np.random.randn(n, k)
    B = np.random.randn(k, n)
    fast_estimate = self._help_product_norm_fast(A, B)
    exact_value = self._help_product_norm_slow(A, B)
    assert_(fast_estimate <= exact_value <= 3 * fast_estimate, f'fast: {fast_estimate:g}\nexact:{exact_value:g}')