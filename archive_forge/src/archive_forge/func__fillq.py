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
def _fillq(self):
    """Fill up both Audio and Video queues if space is available in both"""
    self._fillq_scheduled = False
    while len(self.audioq) < self._max_len_audioq and len(self.videoq) < self._max_len_videoq:
        if self._get_packet():
            self._process_packet()
        else:
            self._stream_end = True
            break