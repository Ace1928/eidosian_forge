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
class HIDDeviceElement:

    @classmethod
    def get_element(cls, element_ref):
        return HIDDeviceElement(element_ref)

    def __init__(self, element_ref):
        self.element_ref = element_ref
        self.cookie = iokit.IOHIDElementGetCookie(element_ref)
        self.type = iokit.IOHIDElementGetType(element_ref)
        if self.type == kIOHIDElementTypeCollection:
            self.collectionType = iokit.IOHIDElementGetCollectionType(element_ref)
        else:
            self.collectionType = None
        self.usagePage = iokit.IOHIDElementGetUsagePage(element_ref)
        self.usage = iokit.IOHIDElementGetUsage(element_ref)
        self.isVirtual = bool(iokit.IOHIDElementIsVirtual(element_ref))
        self.isRelative = bool(iokit.IOHIDElementIsRelative(element_ref))
        self.isWrapping = bool(iokit.IOHIDElementIsWrapping(element_ref))
        self.isArray = bool(iokit.IOHIDElementIsArray(element_ref))
        self.isNonLinear = bool(iokit.IOHIDElementIsNonLinear(element_ref))
        self.hasPreferredState = bool(iokit.IOHIDElementHasPreferredState(element_ref))
        self.hasNullState = bool(iokit.IOHIDElementHasNullState(element_ref))
        self.name = cftype_to_value(iokit.IOHIDElementGetName(element_ref))
        self.reportID = iokit.IOHIDElementGetReportID(element_ref)
        self.reportSize = iokit.IOHIDElementGetReportSize(element_ref)
        self.reportCount = iokit.IOHIDElementGetReportCount(element_ref)
        self.unit = iokit.IOHIDElementGetUnit(element_ref)
        self.unitExponent = iokit.IOHIDElementGetUnitExponent(element_ref)
        self.logicalMin = iokit.IOHIDElementGetLogicalMin(element_ref)
        self.logicalMax = iokit.IOHIDElementGetLogicalMax(element_ref)
        self.physicalMin = iokit.IOHIDElementGetPhysicalMin(element_ref)
        self.physicalMax = iokit.IOHIDElementGetPhysicalMax(element_ref)