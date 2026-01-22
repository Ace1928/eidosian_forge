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
@Win32EventHandler(WM_DROPFILES)
def _event_drop_files(self, msg, wParam, lParam):
    drop = wParam
    file_count = _shell32.DragQueryFileW(drop, 4294967295, None, 0)
    point = POINT()
    _shell32.DragQueryPoint(drop, ctypes.byref(point))
    paths = []
    for i in range(file_count):
        length = _shell32.DragQueryFileW(drop, i, None, 0)
        buffer = create_unicode_buffer(length + 1)
        _shell32.DragQueryFileW(drop, i, buffer, length + 1)
        paths.append(buffer.value)
    _shell32.DragFinish(drop)
    self.dispatch_event('on_file_drop', point.x, self._height - point.y, paths)
    return 0