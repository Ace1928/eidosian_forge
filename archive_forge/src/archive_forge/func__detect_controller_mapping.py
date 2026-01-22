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
def _detect_controller_mapping(device):
    mapping = dict(guid=device.get_guid(), name=device.name)
    _aliases = {BTN_MODE: 'guide', BTN_SELECT: 'back', BTN_START: 'start', BTN_SOUTH: 'a', BTN_EAST: 'b', BTN_WEST: 'x', BTN_NORTH: 'y', BTN_TL: 'leftshoulder', BTN_TR: 'rightshoulder', BTN_TL2: 'lefttrigger', BTN_TR2: 'righttrigger', BTN_THUMBL: 'leftstick', BTN_THUMBR: 'rightstick', BTN_DPAD_UP: 'dpup', BTN_DPAD_DOWN: 'dpdown', BTN_DPAD_LEFT: 'dpleft', BTN_DPAD_RIGHT: 'dpright', ABS_HAT0X: 'dpleft', ABS_HAT0Y: 'dpup', ABS_Z: 'lefttrigger', ABS_RZ: 'righttrigger', ABS_X: 'leftx', ABS_Y: 'lefty', ABS_RX: 'rightx', ABS_RY: 'righty'}
    button_controls = [control for control in device.controls if isinstance(control, Button)]
    axis_controls = [control for control in device.controls if isinstance(control, AbsoluteAxis)]
    hat_controls = [control for control in device.controls if control.name in ('hat_x', 'hat_y')]
    for i, control in enumerate(button_controls):
        name = _aliases.get(control._event_code)
        if name:
            mapping[name] = Relation('button', i)
    for i, control in enumerate(axis_controls):
        name = _aliases.get(control._event_code)
        if name:
            mapping[name] = Relation('axis', i)
    for i, control in enumerate(hat_controls):
        name = _aliases.get(control._event_code)
        if name:
            index = 1 + i << 1
            mapping[name] = Relation('hat0', index)
    return mapping