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
@pytest.mark.parametrize('channel_axis', [0, 1, 2, -1])
@pytest.mark.parametrize('convert2ycbcr', [False, True])
def test_wavelet_denoising_channel_axis(channel_axis, convert2ycbcr):
    rstate = np.random.default_rng(1234)
    sigma = 0.1
    img = astro_odd
    noisy = img + sigma * rstate.standard_normal(img.shape)
    noisy = np.clip(noisy, 0, 1)
    img = np.moveaxis(img, -1, channel_axis)
    noisy = np.moveaxis(noisy, -1, channel_axis)
    denoised = restoration.denoise_wavelet(noisy, sigma=sigma, channel_axis=channel_axis, convert2ycbcr=convert2ycbcr, rescale_sigma=True)
    psnr_noisy = peak_signal_noise_ratio(img, noisy)
    psnr_denoised = peak_signal_noise_ratio(img, denoised)
    assert psnr_denoised > psnr_noisy