from ctypes import *
import pyglet
from pyglet.window import BaseWindow
from pyglet.window import MouseCursor, DefaultMouseCursor
from pyglet.window import WindowException
from pyglet.event import EventDispatcher
from pyglet.canvas.cocoa import CocoaCanvas
from pyglet.libs.darwin import cocoapy, CGPoint, AutoReleasePool
from .systemcursor import SystemCursor
from .pyglet_delegate import PygletDelegate
from .pyglet_window import PygletWindow, PygletToolWindow
from .pyglet_view import PygletView
def _set_nice_window_location(self):
    visible_windows = [win for win in pyglet.app.windows if win is not self and win._nswindow and win._nswindow.isVisible()]
    if not visible_windows:
        self._center_window()
    else:
        point = visible_windows[-1]._nswindow.cascadeTopLeftFromPoint_(cocoapy.NSZeroPoint)
        self._nswindow.cascadeTopLeftFromPoint_(point)