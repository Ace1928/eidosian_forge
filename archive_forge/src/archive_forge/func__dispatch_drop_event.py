from os.path import join
import sys
from typing import Optional
from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import WindowBase
from kivy.input.provider import MotionEventProvider
from kivy.input.motionevent import MotionEvent
from kivy.resources import resource_find
from kivy.utils import platform, deprecated
from kivy.compat import unichr
from collections import deque
def _dispatch_drop_event(self, action, args):
    x, y = (0, 0) if self._drop_pos is None else self._drop_pos
    if action == 'dropfile':
        self.dispatch('on_drop_file', args[0], x, y)
    elif action == 'droptext':
        self.dispatch('on_drop_text', args[0], x, y)
    elif action == 'dropbegin':
        self._drop_pos = x, y = self._win.get_relative_mouse_pos()
        self._collide_and_dispatch_cursor_enter(x, y)
        self.dispatch('on_drop_begin', x, y)
    elif action == 'dropend':
        self._drop_pos = None
        self.dispatch('on_drop_end', x, y)