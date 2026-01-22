from __future__ import absolute_import
import ctypes
from serial.tools import list_ports_common
def IOObjectGetClass(device):
    classname = ctypes.create_string_buffer(io_name_size)
    iokit.IOObjectGetClass(device, ctypes.byref(classname))
    return classname.value