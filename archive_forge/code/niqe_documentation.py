from ..utils import *
import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
Computes Naturalness Image Quality Evaluator. [#f1]_

    Input a video of any quality and get back its distance frame-by-frame
    from naturalness.

    Parameters
    ----------
    inputVideoData : ndarray
        Input video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    Returns
    -------
    niqe_array : ndarray
        The niqe results, ndarray of dimension (T,), where T
        is the number of frames

    References
    ----------

    .. [#f1] Mittal, Anish, Rajiv Soundararajan, and Alan C. Bovik. "Making a 'completely blind' image quality analyzer." IEEE Signal Processing Letters 20.3 (2013): 209-212.

    