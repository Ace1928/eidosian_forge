import functools
from math import ceil
import numbers
import scipy.stats
import numpy as np
from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.utils import _supported_float_type, warn
from ._denoise_cy import _denoise_bilateral, _denoise_tv_bregman
from .. import color
from ..color.colorconv import ycbcr_from_rgb
def _wavelet_threshold(image, wavelet, method=None, threshold=None, sigma=None, mode='soft', wavelet_levels=None):
    """Perform wavelet thresholding.

    Parameters
    ----------
    image : ndarray (2d or 3d) of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    wavelet : string
        The type of wavelet to perform. Can be any of the options
        pywt.wavelist outputs. For example, this may be any of ``{db1, db2,
        db3, db4, haar}``.
    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to be used. The currently supported methods are
        "BayesShrink" [1]_ and "VisuShrink" [2]_. If it is set to None, a
        user-specified ``threshold`` must be supplied instead.
    threshold : float, optional
        The thresholding value to apply during wavelet coefficient
        thresholding. The default value (None) uses the selected ``method`` to
        estimate appropriate threshold(s) for noise removal.
    sigma : float, optional
        The standard deviation of the noise. The noise is estimated when sigma
        is None (the default) by the method in [2]_.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.
    wavelet_levels : int or None, optional
        The number of wavelet decomposition levels to use.  The default is
        three less than the maximum number of possible decomposition levels
        (see Notes below).

    Returns
    -------
    out : ndarray
        Denoised image.

    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           :DOI:`10.1109/83.862633`
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           :DOI:`10.1093/biomet/81.3.425`
    """
    try:
        import pywt
    except ImportError:
        raise ImportError('PyWavelets is not installed. Please ensure it is installed in order to use this function.')
    wavelet = pywt.Wavelet(wavelet)
    if not wavelet.orthogonal:
        warn(f'Wavelet thresholding was designed for use with orthogonal wavelets. For nonorthogonal wavelets such as {wavelet.name},results are likely to be suboptimal.')
    original_extent = tuple((slice(s) for s in image.shape))
    if wavelet_levels is None:
        wavelet_levels = pywt.dwtn_max_level(image.shape, wavelet)
        wavelet_levels = max(wavelet_levels - 3, 1)
    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    dcoeffs = coeffs[1:]
    if sigma is None:
        detail_coeffs = dcoeffs[-1]['d' * image.ndim]
        sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')
    if method is not None and threshold is not None:
        warn(f'Thresholding method {method} selected. The user-specified threshold will be ignored.')
    if threshold is None:
        var = sigma ** 2
        if method is None:
            raise ValueError('If method is None, a threshold must be provided.')
        elif method == 'BayesShrink':
            threshold = [{key: _bayes_thresh(level[key], var) for key in level} for level in dcoeffs]
        elif method == 'VisuShrink':
            threshold = _universal_thresh(image, sigma)
        else:
            raise ValueError(f'Unrecognized method: {method}')
    if np.isscalar(threshold):
        denoised_detail = [{key: pywt.threshold(level[key], value=threshold, mode=mode) for key in level} for level in dcoeffs]
    else:
        denoised_detail = [{key: pywt.threshold(level[key], value=thresh[key], mode=mode) for key in level} for thresh, level in zip(threshold, dcoeffs)]
    denoised_coeffs = [coeffs[0]] + denoised_detail
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]