import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_output_empty(self):
    result = graycomatrix(self.image, [10], [0], 4)
    np.testing.assert_array_equal(result[:, :, 0, 0], np.zeros((4, 4), dtype=np.uint32))
    result = graycomatrix(self.image, [10], [0], 4, normed=True)
    np.testing.assert_array_equal(result[:, :, 0, 0], np.zeros((4, 4), dtype=np.uint32))