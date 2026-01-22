import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
class FlatEnvelope(_Envelope):
    """A flat envelope, providing basic amplitude setting.

    :Parameters:
        `amplitude` : float
            The amplitude (volume) of the wave, from 0.0 to 1.0.
            Values outside this range will be clamped.
    """

    def __init__(self, amplitude=0.5):
        self.amplitude = max(min(1.0, amplitude), 0)

    def get_generator(self, sample_rate, duration):
        amplitude = self.amplitude
        while True:
            yield amplitude