import numpy as np
import pytest
from skimage.morphology import flood, flood_fill
def test_overrange_tolerance_float():
    max_value = np.finfo(np.float32).max
    image = np.random.uniform(size=(64, 64), low=-1.0, high=1.0).astype(np.float32)
    image *= max_value
    expected = np.ones_like(image)
    output = flood_fill(image, (0, 1), 1.0, tolerance=max_value.item() * 10)
    np.testing.assert_equal(output, expected)