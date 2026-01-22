import time
import weakref
import threading
import pyglet
from pyglet.libs.win32 import com
from pyglet.event import EventDispatcher
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _ole32 as ole32, _oleaut32 as oleaut32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.input.base import Device, Controller, Button, AbsoluteAxis, ControllerManager
def _initialize_controls(self):
    for button_name in controller_api_to_pyglet.values():
        control = self.device.controls[button_name]
        self._button_controls.append(control)
        self._add_button(control, button_name)
    for axis_name in ('leftx', 'lefty', 'rightx', 'righty', 'lefttrigger', 'righttrigger'):
        control = self.device.controls[axis_name]
        self._axis_controls.append(control)
        self._add_axis(control, axis_name)