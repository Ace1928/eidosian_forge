import numpy as np
from numpy.testing import assert_allclose
from pytest import approx
from sklearn.utils.stats import _weighted_percentile
def test_weighted_median_integer_weights():
    rng = np.random.RandomState(0)
    x = rng.randint(20, size=10)
    weights = rng.choice(5, size=10)
    x_manual = np.repeat(x, weights)
    median = np.median(x_manual)
    w_median = _weighted_percentile(x, weights)
    assert median == approx(w_median)