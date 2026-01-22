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
class DarwinControllerManager(ControllerManager):

    def __init__(self, display=None):
        self._controllers = {}
        for device in _hid_manager.devices:
            if (controller := _create_controller(device, display)):
                self._controllers[device] = controller

        @_hid_manager.event
        def on_connect(hiddevice):
            if (_controller := _create_controller(hiddevice, display)):
                self._controllers[hiddevice] = _controller
                pyglet.app.platform_event_loop.post_event(self, 'on_connect', _controller)

        @_hid_manager.event
        def on_disconnect(hiddevice):
            if hiddevice in self._controllers:
                _controller = self._controllers[hiddevice]
                del self._controllers[hiddevice]
                pyglet.app.platform_event_loop.post_event(self, 'on_disconnect', _controller)

    def get_controllers(self):
        return list(self._controllers.values())