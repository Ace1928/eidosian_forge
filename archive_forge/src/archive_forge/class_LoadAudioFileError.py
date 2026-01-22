from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
from ..processors import BufferProcessor, Processor
from ..utils import integer_types
class LoadAudioFileError(Exception):
    """
    Deprecated as of version 0.16. Please use
    madmom.io.audio.LoadAudioFileError instead. Will be removed in version
    0.18.

    """

    def __init__(self, value=None):
        warnings.warn(LoadAudioFileError.__doc__)
        if value is None:
            value = 'Could not load audio file.'
        self.value = value