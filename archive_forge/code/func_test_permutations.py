import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('random_state', [np.random.RandomState, np.random.default_rng])
@pytest.mark.parametrize('permutation_type, exact_size', [('pairings', special.factorial(3) ** 2), ('samples', 2 ** 3), ('independent', special.binom(6, 3))])
def test_permutations(self, permutation_type, exact_size, random_state):
    x = self.rng.random(3)
    y = self.rng.random(3)

    def statistic(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)
    kwds = {'permutation_type': permutation_type, 'vectorized': True}
    res = stats.permutation_test((x, y), statistic, n_resamples=3, random_state=random_state(0), **kwds)
    assert_equal(res.null_distribution.size, 3)
    res = stats.permutation_test((x, y), statistic, **kwds)
    assert_equal(res.null_distribution.size, exact_size)