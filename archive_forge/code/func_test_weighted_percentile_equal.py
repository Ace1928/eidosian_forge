import numpy as np
from numpy.testing import assert_allclose
from pytest import approx
from sklearn.utils.stats import _weighted_percentile
def test_weighted_percentile_equal():
    y = np.empty(102, dtype=np.float64)
    y.fill(0.0)
    sw = np.ones(102, dtype=np.float64)
    sw[-1] = 0.0
    score = _weighted_percentile(y, sw, 50)
    assert score == 0