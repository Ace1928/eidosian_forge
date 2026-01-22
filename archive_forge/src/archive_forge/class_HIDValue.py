import sys
import warnings
from ctypes import CFUNCTYPE, byref, c_void_p, c_int, c_ubyte, c_bool, c_uint32, c_uint64
import pyglet
from pyglet.event import EventDispatcher
from pyglet.input.base import Device, AbsoluteAxis, RelativeAxis, Button
from pyglet.input.base import Joystick, Controller, AppleRemote, ControllerManager
from pyglet.input.controller import get_mapping, create_guid
from pyglet.libs.darwin.cocoapy import CFSTR, CFIndex, CFTypeID, known_cftypes
from pyglet.libs.darwin.cocoapy import kCFRunLoopDefaultMode, CFAllocatorRef, cf
from pyglet.libs.darwin.cocoapy import cfset_to_set, cftype_to_value, cfarray_to_list
class HIDValue:

    def __init__(self, value_ref):
        assert value_ref
        assert cf.CFGetTypeID(value_ref) == iokit.IOHIDValueGetTypeID()
        self.value_ref = value_ref
        self.timestamp = iokit.IOHIDValueGetTimeStamp(value_ref)
        self.length = iokit.IOHIDValueGetLength(value_ref)
        if self.length <= 4:
            self.intvalue = iokit.IOHIDValueGetIntegerValue(value_ref)
        else:
            self.intvalue = None
        element_ref = c_void_p(iokit.IOHIDValueGetElement(value_ref))
        self.element = HIDDeviceElement.get_element(element_ref)