import numpy as np
from numpy.testing import assert_allclose
from pytest import approx
from sklearn.utils.stats import _weighted_percentile
def test_weighted_percentile_2d():
    rng = np.random.RandomState(0)
    x1 = rng.randint(10, size=10)
    w1 = rng.choice(5, size=10)
    x2 = rng.randint(20, size=10)
    x_2d = np.vstack((x1, x2)).T
    w_median = _weighted_percentile(x_2d, w1)
    p_axis_0 = [_weighted_percentile(x_2d[:, i], w1) for i in range(x_2d.shape[1])]
    assert_allclose(w_median, p_axis_0)
    w2 = rng.choice(5, size=10)
    w_2d = np.vstack((w1, w2)).T
    w_median = _weighted_percentile(x_2d, w_2d)
    p_axis_0 = [_weighted_percentile(x_2d[:, i], w_2d[:, i]) for i in range(x_2d.shape[1])]
    assert_allclose(w_median, p_axis_0)