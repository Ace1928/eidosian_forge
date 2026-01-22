from __future__ import absolute_import, division, print_function
import inspect
import numpy as np
from ..processors import Processor, SequentialProcessor, BufferProcessor
from .filters import (Filterbank, LogarithmicFilterbank, NUM_BANDS, FMIN, FMAX,
def _diff_frames(diff_ratio, hop_size, frame_size, window=np.hanning):
    """
    Compute the number of `diff_frames` for the given ratio of overlap.

    Parameters
    ----------
    diff_ratio : float
        Ratio of overlap of windows of two consecutive STFT frames.
    hop_size : int
        Samples between two adjacent frames.
    frame_size : int
        Size of one frames in samples.
    window : numpy ufunc or array
        Window funtion.

    Returns
    -------
    diff_frames : int
        Number of frames to calculate the difference to.

    """
    if hasattr(window, '__call__'):
        window = window(frame_size)
    sample = np.argmax(window > float(diff_ratio) * max(window))
    diff_samples = len(window) / 2 - sample
    return int(max(1, round(diff_samples / hop_size)))