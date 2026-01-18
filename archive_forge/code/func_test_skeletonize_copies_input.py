import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts
@pytest.mark.parametrize('ndim', [2, 3])
def test_skeletonize_copies_input(self, ndim):
    """Skeletonize mustn't modify the original input image."""
    image = np.ones((3,) * ndim, dtype=bool)
    image = np.pad(image, 1)
    original = image.copy()
    skeletonize(image)
    np.testing.assert_array_equal(image, original)