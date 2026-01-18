import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2
def test_onenormest_table_5_t_1(self):
    t = 1
    n = 100
    itmax = 5
    alpha = 1 - 1e-06
    A = -scipy.linalg.inv(np.identity(n) + alpha * np.eye(n, k=1))
    first_col = np.array([1] + [0] * (n - 1))
    first_row = np.array([(-alpha) ** i for i in range(n)])
    B = -scipy.linalg.toeplitz(first_col, first_row)
    assert_allclose(A, B)
    est, v, w, nmults, nresamples = _onenormest_core(B, B.T, t, itmax)
    exact_value = scipy.linalg.norm(B, 1)
    underest_ratio = est / exact_value
    assert_allclose(underest_ratio, 0.05, rtol=0.0001)
    assert_equal(nmults, 11)
    assert_equal(nresamples, 0)
    est_plain = scipy.sparse.linalg.onenormest(B, t=t, itmax=itmax)
    assert_allclose(est, est_plain)