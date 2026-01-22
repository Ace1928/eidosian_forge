import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
class _Envelope:
    """Base class for SynthesisSource amplitude envelopes."""

    def get_generator(self, sample_rate, duration):
        raise NotImplementedError