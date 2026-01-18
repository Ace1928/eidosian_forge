import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
def test_scalar_t(self):
    actual = geometric_slerp([1, 0], [0, 1], 0.5)
    expected = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2], dtype=np.float64)
    assert actual.shape == (2,)
    assert_allclose(actual, expected)