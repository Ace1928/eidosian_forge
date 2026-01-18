import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('fun_name', ['bootstrap', 'permutation_test', 'monte_carlo_test'])
def test_parameter_vectorized(fun_name):
    rng = np.random.default_rng(75245098234592)
    sample = rng.random(size=10)

    def rvs(size):
        return stats.norm.rvs(size=size, random_state=rng)
    fun_options = {'bootstrap': {'data': (sample,), 'random_state': rng, 'method': 'percentile'}, 'permutation_test': {'data': (sample,), 'random_state': rng, 'permutation_type': 'samples'}, 'monte_carlo_test': {'sample': sample, 'rvs': rvs}}
    common_options = {'n_resamples': 100}
    fun = getattr(stats, fun_name)
    options = fun_options[fun_name]
    options.update(common_options)

    def statistic(x, axis):
        assert x.ndim > 1 or np.array_equal(x, sample)
        return np.mean(x, axis=axis)
    fun(statistic=statistic, vectorized=None, **options)
    fun(statistic=statistic, vectorized=True, **options)

    def statistic(x):
        assert x.ndim == 1
        return np.mean(x)
    fun(statistic=statistic, vectorized=None, **options)
    fun(statistic=statistic, vectorized=False, **options)