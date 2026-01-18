import numpy as np
import scipy.fftpack
import scipy.io
import scipy.ndimage
import scipy.stats
from ..utils import *
def viideo_score(videoData, blocksize=(18, 18), blockoverlap=(8, 8), filterlength=7):
    """Computes VIIDEO score. [#f1]_ [#f2]_

    Since this is a referenceless quality algorithm, only 1 video is needed. This function
    provides the score computed by the algorithm.

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
    score : ndarray
        The video quality score

    References
    ----------

    .. [#f1] A. Mittal, M. A. Saad and A. C. Bovik, "VIIDEO Software Release", URL: http://live.ece.utexas.edu/research/quality/VIIDEO_release.zip, 2014.

    .. [#f2] A. Mittal, M. A. Saad and A. C. Bovik, "A 'Completely Blind' Video Integrity Oracle", submitted to IEEE Transactions in Image Processing, 2014.
    """
    features = viideo_features(videoData, blocksize=blocksize, blockoverlap=blockoverlap, filterlength=filterlength)
    features = features.reshape(features.shape[0], -1, features.shape[3])
    n_len, n_blocks, n_feats = features.shape
    n_len -= 1
    gap = n_len / 10
    step_size = np.round(gap / 2.0)
    if step_size < 1:
        step_size = 1
    scores = []
    for itr in range(0, np.int(n_len + 1), np.int(step_size)):
        f1_cum = []
        f2_cum = []
        for itr_param in range(itr, np.int(np.min((itr + gap + 1, n_len)))):
            low_Fr1 = features[itr_param, :, 2:14]
            low_Fr2 = features[itr_param + 1, :, 2:14]
            high_Fr1 = features[itr_param, :, 16:]
            high_Fr2 = features[itr_param + 1, :, 16:]
            vec1 = np.abs(low_Fr1 - low_Fr2)
            vec2 = np.abs(high_Fr1 - high_Fr2)
            if len(f1_cum) == 0:
                f1_cum = vec1
                f2_cum = vec2
            else:
                f1_cum = np.vstack((f1_cum, vec1))
                f2_cum = np.vstack((f2_cum, vec2))
        if len(f1_cum) > 0:
            A = np.zeros(f1_cum.shape[1], dtype=np.float32)
            for i in range(f1_cum.shape[1]):
                if (np.sum(np.abs(f1_cum[:, i])) != 0) & (np.sum(np.abs(f2_cum[:, i])) != 0):
                    A[i] = scipy.stats.pearsonr(f1_cum[:, i], f2_cum[:, i])[0]
            scores.append(np.mean(A))
    change_score = np.abs(scores - np.roll(scores, 1))
    return np.mean(change_score) + np.mean(scores)