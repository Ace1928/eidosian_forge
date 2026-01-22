import ctypes
import pyglet
from pyglet.input.base import Device, DeviceOpenException
from pyglet.input.base import Button, RelativeAxis, AbsoluteAxis
from pyglet.libs.x11 import xlib
from pyglet.util import asstr
@pyglet.window.xlib.XlibEventHandler(-1)
def _event_xinput_proximity_out(self, ev):
    e = ctypes.cast(ctypes.byref(ev), ctypes.POINTER(xi.XProximityNotifyEvent)).contents
    device = self._responders.get(e.deviceid)
    if device is not None:
        device._proximity_out(e)