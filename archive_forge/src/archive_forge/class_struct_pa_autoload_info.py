import ctypes
from ctypes import *
import pyglet.lib
class struct_pa_autoload_info(Structure):
    __slots__ = ['index', 'name', 'type', 'module', 'argument']