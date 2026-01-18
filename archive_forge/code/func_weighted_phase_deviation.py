from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def weighted_phase_deviation(spectrogram):
    """
    Weighted Phase Deviation.

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.

    Returns
    -------
    weighted_phase_deviation : numpy array
        Weighted phase deviation onset detection function.

    References
    ----------
    .. [1] Simon Dixon,
           "Onset Detection Revisited",
           Proceedings of the 9th International Conference on Digital Audio
           Effects (DAFx), 2006.

    """
    phase = spectrogram.stft.phase()
    if np.shape(phase) != np.shape(spectrogram):
        raise ValueError('spectrogram and phase must be of same shape')
    wpd = np.abs(_phase_deviation(phase) * spectrogram)
    return np.asarray(np.mean(wpd, axis=1))