import ctypes
from ctypes import *
from pyglet import app
from pyglet.app.xlib import XlibSelectDevice
from .base import Display, Screen, ScreenMode, Canvas
from . import xlib_vidmoderestore
from pyglet.libs.x11 import xlib
def _error_handler(display, event):
    import pyglet
    if pyglet.options['debug_x11']:
        event = event.contents
        buf = c_buffer(1024)
        xlib.XGetErrorText(display, event.error_code, buf, len(buf))
        print('X11 error:', buf.value)
        print('   serial:', event.serial)
        print('  request:', event.request_code)
        print('    minor:', event.minor_code)
        print(' resource:', event.resourceid)
        import traceback
        print('Python stack trace (innermost last):')
        traceback.print_stack()
    return 0