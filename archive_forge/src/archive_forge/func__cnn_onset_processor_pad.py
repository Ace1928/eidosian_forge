from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def _cnn_onset_processor_pad(data):
    """Pad the data by repeating the first and last frame 7 times."""
    pad_start = np.repeat(data[:1], 7, axis=0)
    pad_stop = np.repeat(data[-1:], 7, axis=0)
    return np.concatenate((pad_start, data, pad_stop))