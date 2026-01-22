import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.ndimage import correlate
from skimage import draw
from skimage._shared.testing import fetch
from skimage.io import imread
from skimage.morphology import medial_axis, skeletonize, thin
from skimage.morphology._skeletonize import G123_LUT, G123P_LUT, _generate_thin_luts
class TestThin:

    @property
    def input_image(self):
        """image to test thinning with"""
        ii = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 0], [0, 1, 0, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 6, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=float)
        return ii

    def test_zeros(self):
        image = np.zeros((10, 10), dtype=bool)
        assert np.all(thin(image) == False)

    @pytest.mark.parametrize('dtype', [bool, float, int])
    def test_iter_1(self, dtype):
        image = self.input_image.astype(dtype)
        result = thin(image, 1).astype(bool)
        expected = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize('dtype', [bool, float, int])
    def test_noiter(self, dtype):
        image = self.input_image.astype(dtype)
        result = thin(image).astype(bool)
        expected = np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=bool)
        assert_array_equal(result, expected)

    def test_baddim(self):
        for ii in [np.zeros(3, dtype=bool), np.zeros((3, 3, 3), dtype=bool)]:
            with pytest.raises(ValueError):
                thin(ii)

    def test_lut_generation(self):
        g123, g123p = _generate_thin_luts()
        assert_array_equal(g123, G123_LUT)
        assert_array_equal(g123p, G123P_LUT)