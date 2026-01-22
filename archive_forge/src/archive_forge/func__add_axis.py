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
def _add_axis(self, control, name):
    tscale = 1.0 / (control.max - control.min)
    scale = 2.0 / (control.max - control.min)
    bias = -1.0 - control.min * scale
    if name in ('lefttrigger', 'righttrigger'):

        @control.event
        def on_change(value):
            normalized_value = value * tscale
            setattr(self, name, normalized_value)
            self.dispatch_event('on_trigger_motion', self, name, normalized_value)
    elif name in ('leftx', 'lefty'):

        @control.event
        def on_change(value):
            normalized_value = value * scale + bias
            setattr(self, name, normalized_value)
            self.dispatch_event('on_stick_motion', self, 'leftstick', self.leftx, self.lefty)
    elif name in ('rightx', 'righty'):

        @control.event
        def on_change(value):
            normalized_value = value * scale + bias
            setattr(self, name, normalized_value)
            self.dispatch_event('on_stick_motion', self, 'rightstick', self.rightx, self.righty)