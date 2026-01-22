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
def _recheck_devices(self):
    new_devices, missing_devices = self._get_devices()
    if new_devices:
        self.devices.extend(new_devices)
        for device in new_devices:
            self.dispatch_event('on_connect', device)
    if missing_devices:
        for device in missing_devices:
            self.devices.remove(device)
            self.dispatch_event('on_disconnect', device)