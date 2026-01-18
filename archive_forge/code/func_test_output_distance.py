import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_output_distance(self):
    im = np.array([[0, 0, 0, 0], [1, 0, 0, 1], [2, 0, 0, 2], [3, 0, 0, 3]], dtype=np.uint8)
    result = graycomatrix(im, [3], [0], 4, symmetric=False)
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint32)
    np.testing.assert_array_equal(result[:, :, 0, 0], expected)