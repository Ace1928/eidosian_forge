import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
def test_wraparound():
    test = np.zeros((5, 7), dtype=np.float64)
    test[:, 3] = 100
    expected = np.array([[-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, 100.0, 0.0, 0.0, 0.0]])
    np.testing.assert_equal(flood_fill(test, (0, 0), -1), expected)