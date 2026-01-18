import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_output_symmetric_1(self):
    result = graycomatrix(self.image, [1], [np.pi / 2], 4, symmetric=True)
    assert result.shape == (4, 4, 1, 1)
    expected = np.array([[6, 0, 2, 0], [0, 4, 2, 0], [2, 2, 2, 2], [0, 0, 2, 0]], dtype=np.uint32)
    np.testing.assert_array_equal(result[:, :, 0, 0], expected)