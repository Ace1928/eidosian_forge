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
def test_wavelet_denoising_args(rescale_sigma):
    """
    Some of the functions inside wavelet denoising throw an error the wrong
    arguments are passed. This protects against that and verifies that all
    arguments can be passed.
    """
    img = astro
    noisy = img.copy() + 0.1 * np.random.standard_normal(img.shape)
    for convert2ycbcr in [True, False]:
        for multichannel in [True, False]:
            channel_axis = -1 if multichannel else None
            if convert2ycbcr and (not multichannel):
                with pytest.raises(ValueError):
                    restoration.denoise_wavelet(noisy, convert2ycbcr=convert2ycbcr, channel_axis=channel_axis, rescale_sigma=rescale_sigma)
                continue
            for sigma in [0.1, [0.1, 0.1, 0.1], None]:
                if not multichannel and (not convert2ycbcr) or (isinstance(sigma, list) and (not multichannel)):
                    continue
                restoration.denoise_wavelet(noisy, sigma=sigma, convert2ycbcr=convert2ycbcr, channel_axis=channel_axis, rescale_sigma=rescale_sigma)