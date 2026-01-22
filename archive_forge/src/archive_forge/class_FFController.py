import os
import time
import fcntl
import ctypes
import warnings
from ctypes import c_uint16 as _u16
from ctypes import c_int16 as _s16
from ctypes import c_uint32 as _u32
from ctypes import c_int32 as _s32
from ctypes import c_int64 as _s64
from concurrent.futures import ThreadPoolExecutor
from typing import List
import pyglet
from .evdev_constants import *
from pyglet.app.xlib import XlibSelectDevice
from pyglet.input.base import Device, RelativeAxis, AbsoluteAxis, Button, Joystick, Controller
from pyglet.input.base import DeviceOpenException, ControllerManager
from pyglet.input.controller import get_mapping, Relation, create_guid
class FFController(Controller):
    """Controller that supports force-feedback"""
    _fileno = None
    _weak_effect = None
    _play_weak_event = None
    _stop_weak_event = None
    _strong_effect = None
    _play_strong_event = None
    _stop_strong_event = None

    def open(self, window=None, exclusive=False):
        super().open(window, exclusive)
        self._fileno = self.device.fileno()
        self._weak_effect = FFEvent(FF_RUMBLE, -1)
        EVIOCSFF(self._fileno, self._weak_effect)
        self._play_weak_event = InputEvent(Timeval(), EV_FF, self._weak_effect.id, 1)
        self._stop_weak_event = InputEvent(Timeval(), EV_FF, self._weak_effect.id, 0)
        self._strong_effect = FFEvent(FF_RUMBLE, -1)
        EVIOCSFF(self._fileno, self._strong_effect)
        self._play_strong_event = InputEvent(Timeval(), EV_FF, self._strong_effect.id, 1)
        self._stop_strong_event = InputEvent(Timeval(), EV_FF, self._strong_effect.id, 0)

    def rumble_play_weak(self, strength=1.0, duration=0.5):
        effect = self._weak_effect
        effect.u.ff_rumble_effect.weak_magnitude = int(max(min(1.0, strength), 0) * 65535)
        effect.ff_replay.length = int(duration * 1000)
        EVIOCSFF(self._fileno, effect)
        self.device.ff_upload_effect(self._play_weak_event)

    def rumble_play_strong(self, strength=1.0, duration=0.5):
        effect = self._strong_effect
        effect.u.ff_rumble_effect.strong_magnitude = int(max(min(1.0, strength), 0) * 65535)
        effect.ff_replay.length = int(duration * 1000)
        EVIOCSFF(self._fileno, effect)
        self.device.ff_upload_effect(self._play_strong_event)

    def rumble_stop_weak(self):
        self.device.ff_upload_effect(self._stop_weak_event)

    def rumble_stop_strong(self):
        self.device.ff_upload_effect(self._stop_strong_event)