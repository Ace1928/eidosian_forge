import numpy as np
import scipy.fftpack
import scipy.io
import scipy.ndimage
import scipy.stats
from ..utils import *
Computes VIIDEO features. [#f1]_ [#f2]_

    Since this is a referenceless quality algorithm, only 1 video is needed. This function
    provides the raw features used by the algorithm.

    Parameters
    ----------
    videoData : ndarray
        Reference video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels.

    blocksize : tuple (2,)
      
    blockoverlap: tuple (2,)

    Returns
    -------
    features : ndarray
        The individual features of the algorithm.

    References
    ----------

    .. [#f1] A. Mittal, M. A. Saad and A. C. Bovik, "VIIDEO Software Release", URL: http://live.ece.utexas.edu/research/quality/VIIDEO_release.zip, 2014.

    .. [#f2] A. Mittal, M. A. Saad and A. C. Bovik, "A 'Completely Blind' Video Integrity Oracle", submitted to IEEE Transactions in Image Processing, 2014.

    