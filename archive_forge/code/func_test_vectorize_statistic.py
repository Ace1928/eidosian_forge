import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_vectorize_statistic(axis):

    def statistic(*data, axis):
        return sum((sample.mean(axis) for sample in data))

    def statistic_1d(*data):
        for sample in data:
            assert sample.ndim == 1
        return statistic(*data, axis=0)
    statistic2 = _resampling._vectorize_statistic(statistic_1d)
    np.random.seed(0)
    x = np.random.rand(4, 5, 6)
    y = np.random.rand(4, 1, 6)
    z = np.random.rand(1, 5, 6)
    res1 = statistic(x, y, z, axis=axis)
    res2 = statistic2(x, y, z, axis=axis)
    assert_allclose(res1, res2)