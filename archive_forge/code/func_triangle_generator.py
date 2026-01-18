import math as _math
import struct as _struct
from random import uniform as _uniform
from pyglet.media.codecs.base import Source, AudioFormat, AudioData
def triangle_generator(frequency, sample_rate):
    step = 4 * frequency / sample_rate
    value = 0
    while True:
        if value > 1:
            value = 1 - (value - 1)
            step = -step
        if value < -1:
            value = -1 - (value - -1)
            step = -step
        yield value
        value += step