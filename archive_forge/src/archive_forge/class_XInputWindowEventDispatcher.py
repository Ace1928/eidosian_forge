import ctypes
import pyglet
from pyglet.input.base import Device, DeviceOpenException
from pyglet.input.base import Button, RelativeAxis, AbsoluteAxis
from pyglet.libs.x11 import xlib
from pyglet.util import asstr
class XInputWindowEventDispatcher:

    def __init__(self, window):
        self.window = window
        self._responders = {}

    @staticmethod
    def get_dispatcher(window):
        try:
            dispatcher = window.__xinput_window_event_dispatcher
        except AttributeError:
            dispatcher = window.__xinput_window_event_dispatcher = XInputWindowEventDispatcher(window)
        return dispatcher

    def set_responder(self, device_id, responder):
        self._responders[device_id] = responder

    def remove_responder(self, device_id):
        del self._responders[device_id]

    def open_device(self, device_id, device, responder):
        self.set_responder(device_id, responder)
        device = device.contents
        if not device.num_classes:
            return
        events = []

        def add(class_info, event, handler):
            _type = class_info.event_type_base + event
            _class = device_id << 8 | _type
            events.append(_class)
            self.window._event_handlers[_type] = handler
        for i in range(device.num_classes):
            class_info = device.classes[i]
            if class_info.input_class == xi.KeyClass:
                add(class_info, xi._deviceKeyPress, self._event_xinput_key_press)
                add(class_info, xi._deviceKeyRelease, self._event_xinput_key_release)
            elif class_info.input_class == xi.ButtonClass:
                add(class_info, xi._deviceButtonPress, self._event_xinput_button_press)
                add(class_info, xi._deviceButtonRelease, self._event_xinput_button_release)
            elif class_info.input_class == xi.ValuatorClass:
                add(class_info, xi._deviceMotionNotify, self._event_xinput_motion)
            elif class_info.input_class == xi.ProximityClass:
                add(class_info, xi._proximityIn, self._event_xinput_proximity_in)
                add(class_info, xi._proximityOut, self._event_xinput_proximity_out)
            elif class_info.input_class == xi.FeedbackClass:
                pass
            elif class_info.input_class == xi.FocusClass:
                pass
            elif class_info.input_class == xi.OtherClass:
                pass
        array = (xi.XEventClass * len(events))(*events)
        xi.XSelectExtensionEvent(self.window._x_display, self.window._window, array, len(array))

    @pyglet.window.xlib.XlibEventHandler(0)
    def _event_xinput_key_press(self, ev):
        e = ctypes.cast(ctypes.byref(ev), ctypes.POINTER(xi.XDeviceKeyEvent)).contents
        device = self._responders.get(e.deviceid)
        if device is not None:
            device._key_press(e)

    @pyglet.window.xlib.XlibEventHandler(0)
    def _event_xinput_key_release(self, ev):
        e = ctypes.cast(ctypes.byref(ev), ctypes.POINTER(xi.XDeviceKeyEvent)).contents
        device = self._responders.get(e.deviceid)
        if device is not None:
            device._key_release(e)

    @pyglet.window.xlib.XlibEventHandler(0)
    def _event_xinput_button_press(self, ev):
        e = ctypes.cast(ctypes.byref(ev), ctypes.POINTER(xi.XDeviceButtonEvent)).contents
        device = self._responders.get(e.deviceid)
        if device is not None:
            device._button_press(e)

    @pyglet.window.xlib.XlibEventHandler(0)
    def _event_xinput_button_release(self, ev):
        e = ctypes.cast(ctypes.byref(ev), ctypes.POINTER(xi.XDeviceButtonEvent)).contents
        device = self._responders.get(e.deviceid)
        if device is not None:
            device._button_release(e)

    @pyglet.window.xlib.XlibEventHandler(0)
    def _event_xinput_motion(self, ev):
        e = ctypes.cast(ctypes.byref(ev), ctypes.POINTER(xi.XDeviceMotionEvent)).contents
        device = self._responders.get(e.deviceid)
        if device is not None:
            device._motion(e)

    @pyglet.window.xlib.XlibEventHandler(0)
    def _event_xinput_proximity_in(self, ev):
        e = ctypes.cast(ctypes.byref(ev), ctypes.POINTER(xi.XProximityNotifyEvent)).contents
        device = self._responders.get(e.deviceid)
        if device is not None:
            device._proximity_in(e)

    @pyglet.window.xlib.XlibEventHandler(-1)
    def _event_xinput_proximity_out(self, ev):
        e = ctypes.cast(ctypes.byref(ev), ctypes.POINTER(xi.XProximityNotifyEvent)).contents
        device = self._responders.get(e.deviceid)
        if device is not None:
            device._proximity_out(e)