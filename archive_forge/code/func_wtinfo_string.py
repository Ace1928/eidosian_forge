import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
def wtinfo_string(category, index):
    size = lib.WTInfoW(category, index, None)
    buffer = ctypes.create_unicode_buffer(size)
    lib.WTInfoW(category, index, buffer)
    return buffer.value