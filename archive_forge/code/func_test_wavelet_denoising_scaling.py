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
@pytest.mark.parametrize('case', ['1d', pytest.param('2d multichannel', marks=xfail_without_pywt)])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int16, np.uint8])
@pytest.mark.parametrize('convert2ycbcr', [True, pytest.param(False, marks=xfail_without_pywt)])
@pytest.mark.parametrize('estimate_sigma', [pytest.param(True, marks=xfail_without_pywt), False])
def test_wavelet_denoising_scaling(case, dtype, convert2ycbcr, estimate_sigma):
    """Test cases for images without prescaling via img_as_float."""
    rstate = np.random.default_rng(1234)
    if case == '1d':
        x = np.linspace(0, 255, 1024)
    elif case == '2d multichannel':
        x = data.astronaut()[:64, :64]
    x = x.astype(dtype)
    sigma = 25.0
    noisy = x + sigma * rstate.standard_normal(x.shape)
    noisy = np.clip(noisy, x.min(), x.max())
    noisy = noisy.astype(x.dtype)
    channel_axis = -1 if x.shape[-1] == 3 else None
    if estimate_sigma:
        sigma_est = restoration.estimate_sigma(noisy, channel_axis=channel_axis)
    else:
        sigma_est = None
    if convert2ycbcr and channel_axis is None:
        with pytest.raises(ValueError):
            denoised = restoration.denoise_wavelet(noisy, sigma=sigma_est, wavelet='sym4', channel_axis=channel_axis, convert2ycbcr=convert2ycbcr, rescale_sigma=True)
        return
    denoised = restoration.denoise_wavelet(noisy, sigma=sigma_est, wavelet='sym4', channel_axis=channel_axis, convert2ycbcr=convert2ycbcr, rescale_sigma=True)
    assert denoised.dtype == _supported_float_type(noisy.dtype)
    data_range = x.max() - x.min()
    psnr_noisy = peak_signal_noise_ratio(x, noisy, data_range=data_range)
    clipped = np.dtype(dtype).kind != 'f'
    if not clipped:
        psnr_denoised = peak_signal_noise_ratio(x, denoised, data_range=data_range)
        assert denoised.max() > 0.9 * x.max()
    else:
        x_as_float = img_as_float(x)
        f_data_range = x_as_float.max() - x_as_float.min()
        psnr_denoised = peak_signal_noise_ratio(x_as_float, denoised, data_range=f_data_range)
        assert denoised.max() <= 1.0
        if np.dtype(dtype).kind == 'u':
            assert denoised.min() >= 0
        else:
            assert denoised.min() >= -1
    assert psnr_denoised > psnr_noisy