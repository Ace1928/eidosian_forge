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
def _event_devicechange(self, msg, wParam, lParam):
    if lParam == 0:
        return
    if wParam == DBT_DEVICEARRIVAL or wParam == DBT_DEVICEREMOVECOMPLETE:
        hdr_ptr = ctypes.cast(lParam, ctypes.POINTER(DEV_BROADCAST_HDR))
        if hdr_ptr.contents.dbch_devicetype == DBT_DEVTYP_DEVICEINTERFACE:
            pyglet.app.platform_event_loop.post_event(self, '_recheck_devices')