import numpy as np
import pytest
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage.feature import (
from skimage.transform import integral_image
def test_nri_uniform(self):
    lbp = local_binary_pattern(self.image, 8, 1, 'nri_uniform')
    ref = np.array([[0, 54, 0, 57, 12, 57], [34, 0, 58, 58, 3, 22], [58, 57, 15, 50, 0, 47], [10, 3, 40, 42, 35, 0], [57, 7, 57, 58, 0, 56], [9, 58, 0, 57, 7, 14]])
    np.testing.assert_array_almost_equal(lbp, ref)