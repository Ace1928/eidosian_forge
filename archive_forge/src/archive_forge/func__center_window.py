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
def _center_window(self):
    x = self.screen.x + int((self.screen.width - self._width) // 2)
    y = self.screen.y + int((self.screen.height - self._height) // 2)
    self._nswindow.setFrameOrigin_(cocoapy.NSPoint(x, y))