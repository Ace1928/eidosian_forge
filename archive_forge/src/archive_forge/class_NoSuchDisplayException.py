import ctypes
from ctypes import *
from pyglet import app
from pyglet.app.xlib import XlibSelectDevice
from .base import Display, Screen, ScreenMode, Canvas
from . import xlib_vidmoderestore
from pyglet.libs.x11 import xlib
class NoSuchDisplayException(Exception):
    pass