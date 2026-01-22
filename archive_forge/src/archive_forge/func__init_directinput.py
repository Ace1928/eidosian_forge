import ctypes
import warnings
from typing import List, Dict, Optional
from pyglet.libs.win32.constants import WM_DEVICECHANGE, DBT_DEVICEARRIVAL, DBT_DEVICEREMOVECOMPLETE, \
from pyglet.event import EventDispatcher
import pyglet
from pyglet.input import base
from pyglet.libs import win32
from pyglet.libs.win32 import dinput, _user32, DEV_BROADCAST_DEVICEINTERFACE, com, DEV_BROADCAST_HDR
from pyglet.libs.win32 import _kernel32
from pyglet.input.controller import get_mapping
from pyglet.input.base import ControllerManager
def _init_directinput():
    _i_dinput = dinput.IDirectInput8()
    module_handle = _kernel32.GetModuleHandleW(None)
    dinput.DirectInput8Create(module_handle, dinput.DIRECTINPUT_VERSION, dinput.IID_IDirectInput8W, ctypes.byref(_i_dinput), None)
    return _i_dinput