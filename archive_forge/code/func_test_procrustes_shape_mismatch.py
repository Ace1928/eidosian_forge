import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from pytest import raises as assert_raises
from scipy.spatial import procrustes
def test_procrustes_shape_mismatch(self):
    assert_raises(ValueError, procrustes, np.array([[1, 2], [3, 4]]), np.array([[5, 6, 7], [8, 9, 10]]))