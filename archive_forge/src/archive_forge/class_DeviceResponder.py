import ctypes
import pyglet
from pyglet.input.base import Device, DeviceOpenException
from pyglet.input.base import Button, RelativeAxis, AbsoluteAxis
from pyglet.libs.x11 import xlib
from pyglet.util import asstr
class DeviceResponder:

    def _key_press(self, e):
        pass

    def _key_release(self, e):
        pass

    def _button_press(self, e):
        pass

    def _button_release(self, e):
        pass

    def _motion(self, e):
        pass

    def _proximity_in(self, e):
        pass

    def _proximity_out(self, e):
        pass