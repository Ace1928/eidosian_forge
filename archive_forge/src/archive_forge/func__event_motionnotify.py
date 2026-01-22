import locale
import unicodedata
import urllib.parse
from ctypes import *
from functools import lru_cache
from typing import Optional
import pyglet
from pyglet.window import WindowException, MouseCursorException
from pyglet.window import MouseCursor, DefaultMouseCursor, ImageMouseCursor
from pyglet.window import BaseWindow, _PlatformEventHandler, _ViewEventHandler
from pyglet.window import key
from pyglet.window import mouse
from pyglet.event import EventDispatcher
from pyglet.canvas.xlib import XlibCanvas
from pyglet.libs.x11 import xlib
from pyglet.libs.x11 import cursorfont
from pyglet.util import asbytes
@XlibEventHandler(xlib.MotionNotify)
def _event_motionnotify(self, ev):
    buttons = 0
    if ev.xmotion.state & xlib.Button1MotionMask:
        buttons |= mouse.LEFT
    if ev.xmotion.state & xlib.Button2MotionMask:
        buttons |= mouse.MIDDLE
    if ev.xmotion.state & xlib.Button3MotionMask:
        buttons |= mouse.RIGHT
    if buttons:
        x = ev.xmotion.x - self._view_x
        y = self._height - (ev.xmotion.y - self._view_y - 1)
        if self._mouse_in_window:
            dx = x - self._mouse_x
            dy = y - self._mouse_y
        else:
            dx = dy = 0
        self._mouse_x = x
        self._mouse_y = y
        modifiers = self._translate_modifiers(ev.xmotion.state)
        self.dispatch_event('on_mouse_drag', x, y, dx, dy, buttons, modifiers)