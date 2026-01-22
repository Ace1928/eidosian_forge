import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
class TestMorphology:

    def _build_expected_output(self):
        footprints_2D = (footprints.square, footprints.diamond, footprints.disk, footprints.star)
        image = img_as_ubyte(transform.downscale_local_mean(color.rgb2gray(data.coffee()), (20, 20)))
        output = {}
        for n in range(1, 4):
            for strel in footprints_2D:
                for func in gray_morphology_funcs:
                    key = f'{strel.__name__}_{n}_{func.__name__}'
                    output[key] = func(image, strel(n))
        return output

    def test_gray_morphology(self):
        expected = dict(np.load(fetch('data/gray_morph_output.npz')))
        calculated = self._build_expected_output()
        assert_equal(expected, calculated)

    def test_gray_closing_extensive(self):
        img = data.coins()
        footprint = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
        result_default = gray.closing(img, footprint=footprint)
        assert not np.all(result_default >= img)
        result = gray.closing(img, footprint=footprint, mode='ignore')
        assert np.all(result >= img)

    def test_gray_opening_anti_extensive(self):
        img = data.coins()
        footprint = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
        result_default = gray.opening(img, footprint=footprint)
        assert not np.all(result_default <= img)
        result_ignore = gray.opening(img, footprint=footprint, mode='ignore')
        assert np.all(result_ignore <= img)

    @pytest.mark.parametrize('func', gray_morphology_funcs)
    @pytest.mark.parametrize('mode', gray._SUPPORTED_MODES)
    def test_supported_mode(self, func, mode):
        img = np.ones((10, 10))
        func(img, mode=mode)

    @pytest.mark.parametrize('func', gray_morphology_funcs)
    @pytest.mark.parametrize('mode', ['', 'symmetric', 3, None])
    def test_unsupported_mode(self, func, mode):
        img = np.ones((10, 10))
        with pytest.raises(ValueError, match='unsupported mode'):
            func(img, mode=mode)