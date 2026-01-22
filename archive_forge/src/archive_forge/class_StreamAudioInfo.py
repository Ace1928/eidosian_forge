import sys
from collections import deque
from ctypes import (c_int, c_int32, c_uint8, c_char_p,
import pyglet
import pyglet.lib
from pyglet import image
from pyglet.util import asbytes, asstr
from . import MediaDecoder
from .base import AudioData, SourceInfo, StaticSource
from .base import StreamingSource, VideoFormat, AudioFormat
from .ffmpeg_lib import *
from ..exceptions import MediaFormatException
class StreamAudioInfo:

    def __init__(self, sample_format, sample_rate, channels):
        self.sample_format = sample_format
        self.sample_rate = sample_rate
        self.sample_bits = None
        self.channels = channels