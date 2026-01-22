import ctypes
from ctypes import *
import pyglet.lib
class struct_timeval(Structure):
    _fields_ = [('tv_sec', c_long), ('tv_usec', c_long)]