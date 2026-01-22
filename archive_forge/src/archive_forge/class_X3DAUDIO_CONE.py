import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class X3DAUDIO_CONE(Structure):
    _fields_ = [('InnerAngle', FLOAT32), ('OuterAngle', FLOAT32), ('InnerVolume', FLOAT32), ('OuterVolume', FLOAT32), ('InnerLPF', FLOAT32), ('OuterLPF', FLOAT32), ('InnerReverb', FLOAT32), ('OuterReverb', FLOAT32)]