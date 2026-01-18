import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
@run_in_parallel()
def test_output_angles(self):
    result = graycomatrix(self.image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], 4)
    assert result.shape == (4, 4, 1, 4)
    expected1 = np.array([[2, 2, 1, 0], [0, 2, 0, 0], [0, 0, 3, 1], [0, 0, 0, 1]], dtype=np.uint32)
    np.testing.assert_array_equal(result[:, :, 0, 0], expected1)
    expected2 = np.array([[1, 1, 3, 0], [0, 1, 1, 0], [0, 0, 0, 2], [0, 0, 0, 0]], dtype=np.uint32)
    np.testing.assert_array_equal(result[:, :, 0, 1], expected2)
    expected3 = np.array([[3, 0, 2, 0], [0, 2, 2, 0], [0, 0, 1, 2], [0, 0, 0, 0]], dtype=np.uint32)
    np.testing.assert_array_equal(result[:, :, 0, 2], expected3)
    expected4 = np.array([[2, 0, 0, 0], [1, 1, 2, 0], [0, 0, 2, 1], [0, 0, 0, 0]], dtype=np.uint32)
    np.testing.assert_array_equal(result[:, :, 0, 3], expected4)