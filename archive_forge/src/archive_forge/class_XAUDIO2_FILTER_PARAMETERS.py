import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class XAUDIO2_FILTER_PARAMETERS(Structure):
    _fields_ = [('Type', XAUDIO2_FILTER_TYPE), ('Frequency', FLOAT), ('OneOverQ', FLOAT)]