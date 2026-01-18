import numpy as np
import scipy.fftpack
import scipy.io
import scipy.ndimage
import scipy.stats
from ..utils import *
def viideo_features(videoData, blocksize=(18, 18), blockoverlap=(8, 8), filterlength=7):
    """Computes VIIDEO features. [#f1]_ [#f2]_

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

    """
    videoData = vshape(videoData)
    T, M, N, C = videoData.shape
    assert C == 1, 'viideo called with video having %d channels. Please supply only the luminance channel.' % (C,)
    hf = gauss_window_full(filterlength, filterlength / 6.0)
    blockstrideY = blocksize[0]
    blockstrideX = blocksize[1]
    Mn = np.int(np.round((M + blockoverlap[0]) / np.float32(blocksize[0])))
    Nn = np.int(np.round((N + blockoverlap[1]) / np.float32(blocksize[1])))
    features = np.zeros((np.int(T / 2), Mn, Nn, 28), dtype=np.float32)
    for k in range(np.int(T / 2)):
        frame1 = videoData[k * 2, :, :, 0].astype(np.float32)
        frame2 = videoData[k * 2 + 1, :, :, 0].astype(np.float32)
        diff = frame1 - frame2
        for itr in range(0, 2):
            mscn, _, mu = compute_image_mscn_transform(diff, avg_window=hf, extend_mode='nearest')
            h, v, d1, d2 = paired_product(mscn)
            top_pad = blockoverlap[0]
            left_pad = blockoverlap[1]
            leftover = M % blocksize[0]
            bot_pad = 0
            if leftover > 0:
                bot_pad = blockoverlap[0] + blocksize[0] - leftover
            leftover = N % blocksize[1]
            right_pad = 0
            if leftover > 0:
                right_pad = blockoverlap[1] + blocksize[1] - leftover
            mscn = np.pad(mscn, ((top_pad, bot_pad), (left_pad, right_pad)), mode='constant')
            h = np.pad(h, ((top_pad, bot_pad), (left_pad, right_pad)), mode='constant')
            v = np.pad(v, ((top_pad, bot_pad), (left_pad, right_pad)), mode='constant')
            d1 = np.pad(d1, ((top_pad, bot_pad), (left_pad, right_pad)), mode='constant')
            d2 = np.pad(d2, ((top_pad, bot_pad), (left_pad, right_pad)), mode='constant')
            blockheight = blocksize[0] + blockoverlap[0] * 2
            blockwidth = blocksize[1] + blockoverlap[1] * 2
            for j in range(Nn):
                for i in range(Mn):
                    yp = i * blocksize[0]
                    xp = j * blocksize[1]
                    patch = mscn[yp:yp + blockheight, xp:xp + blockwidth].copy()
                    ph = h[yp:yp + blockheight, xp:xp + blockwidth].copy()
                    pv = v[yp:yp + blockheight, xp:xp + blockwidth].copy()
                    pd1 = d1[yp:yp + blockheight, xp:xp + blockwidth].copy()
                    pd2 = d2[yp:yp + blockheight, xp:xp + blockwidth].copy()
                    shape, _, bl, br, _, _ = aggd_features(patch)
                    shapeh, _, blh, brh, _, _ = aggd_features(ph)
                    shapev, _, blv, brv, _, _ = aggd_features(pv)
                    shaped1, _, bld1, brd1, _, _ = aggd_features(pd1)
                    shaped2, _, bld2, brd2, _, _ = aggd_features(pd2)
                    features[k, i, j, itr * 14:(itr + 1) * 14] = np.array([shape, (bl + br) / 2.0, shapev, blv, brv, shapeh, blh, brh, shaped1, bld1, brd1, shaped2, bld2, brd2])
            diff = mu
    return features