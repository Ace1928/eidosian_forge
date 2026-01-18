import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.xslow()
@pytest.mark.parametrize('axis', (-2, 1))
def test_vectorized_nsamp_ptype_samples(self, axis):
    x = self.rng.random(size=(2, 4, 3))
    y = self.rng.random(size=(1, 4, 3))
    z = self.rng.random(size=(2, 4, 1))
    x = stats.rankdata(x, axis=axis)
    y = stats.rankdata(y, axis=axis)
    z = stats.rankdata(z, axis=axis)
    y = y[0]
    data = (x, y, z)

    def statistic1d(*data):
        return stats.page_trend_test(data, ranked=True, method='asymptotic').statistic

    def pvalue1d(*data):
        return stats.page_trend_test(data, ranked=True, method='exact').pvalue
    statistic = _resampling._vectorize_statistic(statistic1d)
    pvalue = _resampling._vectorize_statistic(pvalue1d)
    expected_statistic = statistic(*np.broadcast_arrays(*data), axis=axis)
    expected_pvalue = pvalue(*np.broadcast_arrays(*data), axis=axis)
    kwds = {'vectorized': False, 'axis': axis, 'alternative': 'greater', 'permutation_type': 'pairings', 'random_state': 0}
    res = permutation_test(data, statistic1d, n_resamples=np.inf, **kwds)
    res2 = permutation_test(data, statistic1d, n_resamples=5000, **kwds)
    assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
    assert_allclose(res.statistic, res2.statistic, rtol=self.rtol)
    assert_allclose(res.pvalue, expected_pvalue, rtol=self.rtol)
    assert_allclose(res.pvalue, res2.pvalue, atol=0.03)