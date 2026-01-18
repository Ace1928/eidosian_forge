from ..utils import *
from .ssim import *
import numpy as np
import scipy.ndimage
Computes Multiscale Structural Similarity (MS-SSIM) Index. [#f1]_

    Both video inputs are compared frame-by-frame to obtain T
    MS-SSIM measurements on the luminance channel.

    Parameters
    ----------
    referenceVideoData : ndarray
        Reference video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    distortedVideoData : ndarray
        Distorted video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    method : str
        Whether to use "product" (default) or to use "sum" for combing multiple scales into the single score.

    Returns
    -------
    msssim_array : ndarray
        The MS-SSIM results, ndarray of dimension (T,), where T
        is the number of frames

    References
    ----------

    .. [#f1] Z. Wang, E. P. Simoncelli and A. C. Bovik, "Multi-scale structural similarity for image quality assessment," IEEE Asilomar Conference Signals, Systems and Computers, Nov. 2003.

    