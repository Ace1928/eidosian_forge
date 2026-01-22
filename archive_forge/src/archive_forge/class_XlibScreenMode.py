import ctypes
from ctypes import *
from pyglet import app
from pyglet.app.xlib import XlibSelectDevice
from .base import Display, Screen, ScreenMode, Canvas
from . import xlib_vidmoderestore
from pyglet.libs.x11 import xlib
class XlibScreenMode(ScreenMode):

    def __init__(self, screen, info):
        super(XlibScreenMode, self).__init__(screen)
        self.info = info
        self.width = info.hdisplay
        self.height = info.vdisplay
        self.rate = info.dotclock
        self.depth = None