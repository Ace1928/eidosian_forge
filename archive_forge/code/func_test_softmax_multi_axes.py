import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from scipy.special import logsumexp, softmax
def test_softmax_multi_axes():
    assert_allclose(softmax([[1000, 0], [1000, 0]], axis=0), np.array([[0.5, 0.5], [0.5, 0.5]]), rtol=1e-13)
    assert_allclose(softmax([[1000, 0], [1000, 0]], axis=1), np.array([[1, 0], [1, 0]]), rtol=1e-13)
    x = np.array([[-25, 0, 25, 50], [1, 325, 749, 750]])
    expected = np.array([[2.678636961770877e-33, 1.9287498479371314e-22, 1.3887943864771144e-11, 0.999999999986112], [0.0, 1.9444526359919372e-185, 0.2689414213699951, 0.7310585786300048]])
    assert_allclose(softmax(x, axis=1), expected, rtol=1e-13)
    assert_allclose(softmax(x.T, axis=0), expected.T, rtol=1e-13)
    x3d = x.reshape(2, 2, 2)
    assert_allclose(softmax(x3d, axis=(1, 2)), expected.reshape(2, 2, 2), rtol=1e-13)