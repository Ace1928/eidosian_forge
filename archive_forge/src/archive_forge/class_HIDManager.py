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
class HIDManager(EventDispatcher):

    def __init__(self):
        """Create an instance of an HIDManager."""
        self.manager_ref = c_void_p(iokit.IOHIDManagerCreate(None, kIOHIDOptionsTypeNone))
        self.schedule_with_run_loop()
        self.devices = self._get_devices()
        self.matching_callback = self._register_matching_callback()
        self.removal_callback = self._register_removal_callback()

    def _get_devices(self):
        try:
            iokit.IOHIDManagerSetDeviceMatching(self.manager_ref, None)
            cfset = c_void_p(iokit.IOHIDManagerCopyDevices(self.manager_ref))
            devices = cfset_to_set(cfset)
            cf.CFRelease(cfset)
        except:
            return set()
        return devices

    def open(self):
        iokit.IOHIDManagerOpen(self.manager_ref, kIOHIDOptionsTypeNone)

    def close(self):
        iokit.IOHIDManagerClose(self.manager_ref, kIOHIDOptionsTypeNone)

    def schedule_with_run_loop(self):
        iokit.IOHIDManagerScheduleWithRunLoop(self.manager_ref, c_void_p(cf.CFRunLoopGetCurrent()), kCFRunLoopDefaultMode)

    def unschedule_from_run_loop(self):
        iokit.IOHIDManagerUnscheduleFromRunLoop(self.manager_ref, c_void_p(cf.CFRunLoopGetCurrent()), kCFRunLoopDefaultMode)

    def _py_matching_callback(self, context, result, sender, device):
        d = HIDDevice.get_device(c_void_p(device))
        if d not in self.devices:
            self.devices.add(d)
            self.dispatch_event('on_connect', d)

    def _register_matching_callback(self):
        matching_callback = HIDManagerCallback(self._py_matching_callback)
        iokit.IOHIDManagerRegisterDeviceMatchingCallback(self.manager_ref, matching_callback, None)
        return matching_callback

    def _py_removal_callback(self, context, result, sender, device):
        d = HIDDevice.get_device(c_void_p(device))
        d.close()
        if d in self.devices:
            self.devices.remove(d)
            self.dispatch_event('on_disconnect', d)

    def _register_removal_callback(self):
        removal_callback = HIDManagerCallback(self._py_removal_callback)
        iokit.IOHIDManagerRegisterDeviceRemovalCallback(self.manager_ref, removal_callback, None)
        return removal_callback