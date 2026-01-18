import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.slow
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_vector_valued_statistic_gh17715():
    rng = np.random.default_rng(141921000979291141)

    def concordance(x, y, axis):
        xm = x.mean(axis)
        ym = y.mean(axis)
        cov = ((x - xm[..., None]) * (y - ym[..., None])).mean(axis)
        return 2 * cov / (x.var(axis) + y.var(axis) + (xm - ym) ** 2)

    def statistic(tp, tn, fp, fn, axis):
        actual = tp + fp
        expected = tp + fn
        return np.nan_to_num(concordance(actual, expected, axis))

    def statistic_extradim(*args, axis):
        return statistic(*args, axis)[np.newaxis, ...]
    data = [[4, 0, 0, 2], [2, 1, 2, 1], [0, 6, 0, 0], [0, 6, 3, 0], [0, 8, 1, 0]]
    data = np.array(data).T
    res = bootstrap(data, statistic_extradim, random_state=rng, paired=True)
    ref = bootstrap(data, statistic, random_state=rng, paired=True)
    assert_allclose(res.confidence_interval.low[0], ref.confidence_interval.low, atol=1e-15)
    assert_allclose(res.confidence_interval.high[0], ref.confidence_interval.high, atol=1e-15)