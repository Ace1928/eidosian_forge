import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2
@pytest.mark.xslow
def test_onenormest_table_6_t_1(self):
    np.random.seed(1234)
    t = 1
    n = 100
    itmax = 5
    nsamples = 5000
    observed = []
    expected = []
    nmult_list = []
    nresample_list = []
    for i in range(nsamples):
        A_inv = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        A = scipy.linalg.inv(A_inv)
        est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)
        observed.append(est)
        expected.append(scipy.linalg.norm(A, 1))
        nmult_list.append(nmults)
        nresample_list.append(nresamples)
    observed = np.array(observed, dtype=float)
    expected = np.array(expected, dtype=float)
    relative_errors = np.abs(observed - expected) / expected
    underestimation_ratio = observed / expected
    underestimation_ratio_mean = np.mean(underestimation_ratio)
    assert_(0.9 < underestimation_ratio_mean < 0.99)
    max_nresamples = np.max(nresample_list)
    assert_equal(max_nresamples, 0)
    nexact = np.count_nonzero(relative_errors < 1e-14)
    proportion_exact = nexact / float(nsamples)
    assert_(0.7 < proportion_exact < 0.8)
    mean_nmult = np.mean(nmult_list)
    assert_(4 < mean_nmult < 5)