from ..utils import *
import numpy as np
import scipy.ndimage
Computes mean absolute deviation (MAD).

    Both video inputs are compared frame-by-frame to obtain T
    MAD measurements.

    Parameters
    ----------
    referenceVideoData : ndarray
        Reference video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels.

    distortedVideoData : ndarray
        Distorted video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels.

    Returns
    -------
    mad_array : ndarray
        The mad results, ndarray of dimension (T,), where T
        is the number of frames

    