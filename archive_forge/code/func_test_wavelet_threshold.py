import functools
import itertools
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_warns
from skimage import color, data, img_as_float, restoration
from skimage._shared._warnings import expected_warnings
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration._denoise import _wavelet_threshold
@xfail_without_pywt
def test_wavelet_threshold():
    rstate = np.random.default_rng(1234)
    img = astro_gray
    sigma = 0.1
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)
    denoised = _wavelet_threshold(noisy, wavelet='db1', method=None, threshold=sigma)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy
    with pytest.raises(ValueError):
        _wavelet_threshold(noisy, wavelet='db1', method=None, threshold=None)
    with expected_warnings(['Thresholding method ']):
        _wavelet_threshold(noisy, wavelet='db1', method='BayesShrink', threshold=sigma)