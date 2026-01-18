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
@pytest.mark.parametrize('img, channel_axis, convert2ycbcr', [(astro_gray, None, False), (astro_gray_odd, None, False), (astro_odd, -1, False), (astro_odd, -1, True)])
def test_wavelet_denoising(img, channel_axis, convert2ycbcr):
    rstate = np.random.default_rng(1234)
    sigma = 0.1
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)
    denoised = restoration.denoise_wavelet(noisy, sigma=sigma, channel_axis=channel_axis, convert2ycbcr=convert2ycbcr, rescale_sigma=True)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy
    denoised = restoration.denoise_wavelet(noisy, channel_axis=channel_axis, convert2ycbcr=convert2ycbcr, rescale_sigma=True)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy
    denoised_1 = restoration.denoise_wavelet(noisy, channel_axis=channel_axis, wavelet_levels=1, convert2ycbcr=convert2ycbcr, rescale_sigma=True)
    psnr_denoised_1 = peak_signal_noise_ratio(img, denoised_1)
    assert psnr_denoised > psnr_denoised_1
    assert psnr_denoised_1 > psnr_noisy
    res1 = restoration.denoise_wavelet(noisy, sigma=2 * sigma, channel_axis=channel_axis, rescale_sigma=True)
    res2 = restoration.denoise_wavelet(noisy, sigma=sigma, channel_axis=channel_axis, rescale_sigma=True)
    assert np.sum(res1 ** 2) <= np.sum(res2 ** 2)