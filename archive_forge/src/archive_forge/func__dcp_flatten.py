from __future__ import absolute_import, division, print_function
import numpy as np
from madmom.audio.spectrogram import SemitoneBandpassSpectrogram
from madmom.processors import Processor, SequentialProcessor
def _dcp_flatten(fs):
    """Flatten spectrograms for DeepChromaProcessor. Needs to be outside
       of the class in order to be picklable for multiprocessing.
    """
    return np.concatenate(fs).reshape(len(fs), -1)