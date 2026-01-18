import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('method', ['basic', 'percentile', 'BCa'])
def test_vector_valued_statistic(method):
    rng = np.random.default_rng(2196847219)
    params = (1, 0.5)
    sample = stats.norm.rvs(*params, size=(100, 100), random_state=rng)

    def statistic(data, axis):
        return np.asarray([np.mean(data, axis), np.std(data, axis, ddof=1)])
    res = bootstrap((sample,), statistic, method=method, axis=-1, n_resamples=9999, batch=200)
    counts = np.sum((res.confidence_interval.low.T < params) & (res.confidence_interval.high.T > params), axis=0)
    assert np.all(counts >= 90)
    assert np.all(counts <= 100)
    assert res.confidence_interval.low.shape == (2, 100)
    assert res.confidence_interval.high.shape == (2, 100)
    assert res.standard_error.shape == (2, 100)
    assert res.bootstrap_distribution.shape == (2, 100, 9999)