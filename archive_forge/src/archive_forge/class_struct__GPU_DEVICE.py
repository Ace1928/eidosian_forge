from ctypes import *
from pyglet.gl.lib import link_WGL as _link_function
from pyglet.gl.lib import c_void
class struct__GPU_DEVICE(Structure):
    __slots__ = ['cb', 'DeviceName', 'DeviceString', 'Flags', 'rcVirtualScreen']