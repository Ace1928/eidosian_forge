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
@Win32EventHandler(WM_SYSCOMMAND)
def _event_syscommand(self, msg, wParam, lParam):
    if wParam == SC_KEYMENU and lParam & 1 >> 16 <= 0:
        return 0
    if wParam & 65520 in (SC_MOVE, SC_SIZE):
        from pyglet import app
        if app.event_loop is not None:
            app.event_loop.enter_blocking()