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
@pytest.mark.parametrize('rescale_sigma', [True, False])
def test_wavelet_denoising_levels(rescale_sigma):
    rstate = np.random.default_rng(1234)
    ndim = 2
    N = 256
    wavelet = 'db1'
    img = 0.2 * np.ones((N,) * ndim)
    img[(slice(5, 13),) * ndim] = 0.8
    sigma = 0.1
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)
    denoised = restoration.denoise_wavelet(noisy, wavelet=wavelet, rescale_sigma=rescale_sigma)
    denoised_1 = restoration.denoise_wavelet(noisy, wavelet=wavelet, wavelet_levels=1, rescale_sigma=rescale_sigma)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    psnr_denoised_1 = peak_signal_noise_ratio(img, denoised_1)
    assert psnr_denoised > psnr_denoised_1 > psnr_noisy
    max_level = pywt.dwt_max_level(np.min(img.shape), pywt.Wavelet(wavelet).dec_len)
    with expected_warnings(['all coefficients will experience boundary effects']):
        restoration.denoise_wavelet(noisy, wavelet=wavelet, wavelet_levels=max_level + 1, rescale_sigma=rescale_sigma)
    with pytest.raises(ValueError):
        restoration.denoise_wavelet(noisy, wavelet=wavelet, wavelet_levels=-1, rescale_sigma=rescale_sigma)