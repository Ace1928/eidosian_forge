from ctypes import *
from .base import FontException
import pyglet.lib
class FT_UnitVector(Structure):
    _fields_ = [('x', FT_F2Dot14), ('y', FT_F2Dot14)]