import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts
@pytest.mark.parametrize('dtype', [bool, float, int])
def test_vertical_line(self, dtype):
    """Test a thick vertical line, issue #3861"""
    img = np.zeros((9, 9), dtype=dtype)
    img[:, 2] = 1
    img[:, 3] = 2
    img[:, 4] = 3
    expected = np.full(img.shape, False)
    expected[:, 3] = True
    result = medial_axis(img)
    assert_array_equal(result, expected)