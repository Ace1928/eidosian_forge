from ctypes import *
import pyglet.lib
class struct_gbm_device(Structure):
    _fields_ = [('_opaque_struct', c_int)]