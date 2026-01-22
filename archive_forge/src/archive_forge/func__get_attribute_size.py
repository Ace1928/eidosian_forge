import os
import platform
import warnings
from pyglet import image
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32 import com
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.media import Source
from pyglet.media.codecs import AudioFormat, AudioData, VideoFormat, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
@staticmethod
def _get_attribute_size(attributes, guidKey):
    """ Convert int64 attributes to int32"""
    size = ctypes.c_uint64()
    attributes.GetUINT64(guidKey, size)
    lParam = size.value
    x = ctypes.c_int32(lParam).value
    y = ctypes.c_int32(lParam >> 32).value
    return (x, y)