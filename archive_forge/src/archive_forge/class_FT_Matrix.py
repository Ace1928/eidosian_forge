from ctypes import *
from .base import FontException
import pyglet.lib
class FT_Matrix(Structure):
    _fields_ = [('xx', FT_Fixed), ('xy', FT_Fixed), ('yx', FT_Fixed), ('yy', FT_Fixed)]