import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
def wtinfo_uint(category, index):
    buffer = wintab.UINT()
    lib.WTInfoW(category, index, ctypes.byref(buffer))
    return buffer.value