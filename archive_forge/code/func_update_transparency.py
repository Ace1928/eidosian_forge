from ctypes import *
from functools import lru_cache
import unicodedata
from pyglet import compat_platform
import pyglet
from pyglet.window import BaseWindow, WindowException, MouseCursor
from pyglet.window import DefaultMouseCursor, _PlatformEventHandler, _ViewEventHandler
from pyglet.event import EventDispatcher
from pyglet.window import key, mouse
from pyglet.canvas.win32 import Win32Canvas
from pyglet.libs.win32 import _user32, _kernel32, _gdi32, _dwmapi, _shell32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.winkey import *
from pyglet.libs.win32.types import *
def update_transparency(self):
    region = _gdi32.CreateRectRgn(0, 0, -1, -1)
    bb = DWM_BLURBEHIND()
    bb.dwFlags = DWM_BB_ENABLE | DWM_BB_BLURREGION
    bb.hRgnBlur = region
    bb.fEnable = True
    _dwmapi.DwmEnableBlurBehindWindow(self._hwnd, ctypes.byref(bb))
    _gdi32.DeleteObject(region)