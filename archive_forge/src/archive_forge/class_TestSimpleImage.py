import math
import numpy as np
import pytest
from numpy.testing import (
from scipy import ndimage as ndi
from skimage import data, util
from skimage._shared._dependency_checks import has_mpl
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.draw import disk
from skimage.exposure import histogram
from skimage.filters._multiotsu import (
from skimage.filters.thresholding import (
class TestSimpleImage:

    def setup_method(self):
        self.image = np.array([[0, 0, 1, 3, 5], [0, 1, 4, 3, 4], [1, 2, 5, 4, 1], [2, 4, 5, 2, 1], [4, 5, 1, 0, 0]], dtype=int)

    def test_minimum(self):
        with pytest.raises(RuntimeError):
            threshold_minimum(self.image)

    @pytest.mark.skipif(not has_mpl, reason='matplotlib not installed')
    def test_try_all_threshold(self):
        fig, ax = try_all_threshold(self.image)
        all_texts = [axis.texts for axis in ax if axis.texts != []]
        text_content = [text.get_text() for x in all_texts for text in x]
        assert 'RuntimeError' in text_content

    def test_otsu(self):
        assert threshold_otsu(self.image) == 2

    def test_otsu_negative_int(self):
        image = self.image - 2
        assert threshold_otsu(image) == 0

    def test_otsu_float_image(self):
        image = np.float64(self.image)
        assert 2 <= threshold_otsu(image) < 3

    def test_li(self):
        assert 2 < threshold_li(self.image) < 3

    def test_li_negative_int(self):
        image = self.image - 2
        assert 0 < threshold_li(image) < 1

    def test_li_float_image(self):
        image = self.image.astype(float)
        assert 2 < threshold_li(image) < 3

    def test_li_constant_image(self):
        assert threshold_li(np.ones((10, 10))) == 1.0

    def test_yen(self):
        assert threshold_yen(self.image) == 2

    def test_yen_negative_int(self):
        image = self.image - 2
        assert threshold_yen(image) == 0

    def test_yen_float_image(self):
        image = np.float64(self.image)
        assert 2 <= threshold_yen(image) < 3

    def test_yen_arange(self):
        image = np.arange(256)
        assert threshold_yen(image) == 127

    def test_yen_binary(self):
        image = np.zeros([2, 256], dtype=np.uint8)
        image[0] = 255
        assert threshold_yen(image) < 1

    def test_yen_blank_zero(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        assert threshold_yen(image) == 0

    def test_yen_blank_max(self):
        image = np.empty((5, 5), dtype=np.uint8)
        image.fill(255)
        assert threshold_yen(image) == 255

    def test_isodata(self):
        assert threshold_isodata(self.image) == 2
        assert threshold_isodata(self.image, return_all=True) == [2]

    def test_isodata_blank_zero(self):
        image = np.zeros((5, 5), dtype=np.uint8)
        assert threshold_isodata(image) == 0
        assert threshold_isodata(image, return_all=True) == [0]

    def test_isodata_linspace(self):
        image = np.linspace(-127, 0, 256)
        assert -63.8 < threshold_isodata(image) < -63.6
        assert_almost_equal(threshold_isodata(image, return_all=True), [-63.74804688, -63.25195312])

    def test_isodata_16bit(self):
        np.random.seed(0)
        imfloat = np.random.rand(256, 256)
        assert 0.49 < threshold_isodata(imfloat, nbins=1024) < 0.51
        assert all(0.49 < threshold_isodata(imfloat, nbins=1024, return_all=True))

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_threshold_local_gaussian(self, ndim):
        ref = np.array([[False, False, False, False, True], [False, False, True, False, True], [False, False, True, True, False], [False, True, True, False, False], [True, True, False, False, False]])
        if ndim == 2:
            image = self.image
            block_sizes = [3, (3,) * image.ndim]
        else:
            image = np.stack((self.image,) * 5, axis=-1)
            ref = np.stack((ref,) * 5, axis=-1)
            block_sizes = [3, (3,) * image.ndim, (3,) * (image.ndim - 1) + (1,)]
        for block_size in block_sizes:
            out = threshold_local(image, block_size, method='gaussian', mode='reflect')
            assert_equal(ref, image > out)
        out = threshold_local(image, 3, method='gaussian', mode='reflect', param=1 / 3)
        assert_equal(ref, image > out)

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_threshold_local_mean(self, ndim):
        ref = np.array([[False, False, False, False, True], [False, False, True, False, True], [False, False, True, True, False], [False, True, True, False, False], [True, True, False, False, False]])
        if ndim == 2:
            image = self.image
            block_sizes = [3, (3,) * image.ndim]
        else:
            image = np.stack((self.image,) * 5, axis=-1)
            ref = np.stack((ref,) * 5, axis=-1)
            block_sizes = [3, (3,) * image.ndim, (3,) * (image.ndim - 1) + (1,)]
        for block_size in block_sizes:
            out = threshold_local(image, block_size, method='mean', mode='reflect')
            assert_equal(ref, image > out)

    @pytest.mark.parametrize('block_size', [(3,), (3, 3, 3)])
    def test_threshold_local_invalid_block_size(self, block_size):
        with pytest.raises(ValueError):
            threshold_local(self.image, block_size, method='mean')

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_threshold_local_median(self, ndim):
        ref = np.array([[False, False, False, False, True], [False, False, True, False, False], [False, False, True, False, False], [False, False, True, True, False], [False, True, False, False, False]])
        if ndim == 2:
            image = self.image
        else:
            image = np.stack((self.image,) * 5, axis=-1)
            ref = np.stack((ref,) * 5, axis=-1)
        out = threshold_local(image, 3, method='median', mode='reflect')
        assert_equal(ref, image > out)

    def test_threshold_local_median_constant_mode(self):
        out = threshold_local(self.image, 3, method='median', mode='constant', cval=20)
        expected = np.array([[20.0, 1.0, 3.0, 4.0, 20.0], [1.0, 1.0, 3.0, 4.0, 4.0], [2.0, 2.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 1.0, 2.0], [20.0, 5.0, 5.0, 2.0, 20.0]])
        assert_equal(expected, out)

    def test_threshold_niblack(self):
        ref = np.array([[False, False, False, True, True], [False, True, True, True, True], [False, True, True, True, False], [False, True, True, True, True], [True, True, False, False, False]])
        thres = threshold_niblack(self.image, window_size=3, k=0.5)
        out = self.image > thres
        assert_equal(ref, out)

    def test_threshold_sauvola(self):
        ref = np.array([[False, False, False, True, True], [False, False, True, True, True], [False, False, True, True, False], [False, True, True, True, False], [True, True, False, False, False]])
        thres = threshold_sauvola(self.image, window_size=3, k=0.2, r=128)
        out = self.image > thres
        assert_equal(ref, out)

    def test_threshold_niblack_iterable_window_size(self):
        ref = np.array([[False, False, False, True, True], [False, False, True, True, True], [False, True, True, True, False], [False, True, True, True, False], [True, True, False, False, False]])
        thres = threshold_niblack(self.image, window_size=[3, 5], k=0.5)
        out = self.image > thres
        assert_array_equal(ref, out)

    def test_threshold_sauvola_iterable_window_size(self):
        ref = np.array([[False, False, False, True, True], [False, False, True, True, True], [False, False, True, True, False], [False, True, True, True, False], [True, True, False, False, False]])
        thres = threshold_sauvola(self.image, window_size=(3, 5), k=0.2, r=128)
        out = self.image > thres
        assert_array_equal(ref, out)