import ctypes
from ctypes import *
import pyglet.lib
import pyglet.libs.x11.xlib
class struct__XValuatorInfo(Structure):
    __slots__ = ['class', 'length', 'num_axes', 'mode', 'motion_buffer', 'axes']