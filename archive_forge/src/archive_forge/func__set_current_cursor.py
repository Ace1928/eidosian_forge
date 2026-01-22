import ctypes
from collections import defaultdict
import pyglet
from pyglet.input.base import DeviceOpenException
from pyglet.input.base import Tablet, TabletCanvas
from pyglet.libs.win32 import libwintab as wintab
from pyglet.util import debug_print
def _set_current_cursor(self, cursor_type):
    if self._current_cursor:
        self.dispatch_event('on_leave', self._current_cursor)
    self._current_cursor = self.device._cursor_map.get(cursor_type, None)
    if self._current_cursor:
        self.dispatch_event('on_enter', self._current_cursor)