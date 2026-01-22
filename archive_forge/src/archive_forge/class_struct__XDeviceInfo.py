import ctypes
from ctypes import *
import pyglet.lib
import pyglet.libs.x11.xlib
class struct__XDeviceInfo(Structure):
    __slots__ = ['id', 'type', 'name', 'num_classes', 'use', 'inputclassinfo']