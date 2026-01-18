import numpy as np
from numpy.testing import assert_allclose
from pytest import approx
from sklearn.utils.stats import _weighted_percentile
def test_weighted_median_equal_weights():
    rng = np.random.RandomState(0)
    x = rng.randint(10, size=11)
    weights = np.ones(x.shape)
    median = np.median(x)
    w_median = _weighted_percentile(x, weights)
    assert median == approx(w_median)