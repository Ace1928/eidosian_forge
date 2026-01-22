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
@Win32EventHandler(WM_GETMINMAXINFO)
def _event_getminmaxinfo(self, msg, wParam, lParam):
    info = MINMAXINFO.from_address(lParam)
    if self._minimum_size:
        info.ptMinTrackSize.x, info.ptMinTrackSize.y = self._client_to_window_size(*self._minimum_size)
    if self._maximum_size:
        info.ptMaxTrackSize.x, info.ptMaxTrackSize.y = self._client_to_window_size(*self._maximum_size)
    return 0