import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class X3DAUDIO_VECTOR(ctypes.Structure):
    _fields_ = [('x', c_float), ('y', c_float), ('z', c_float)]