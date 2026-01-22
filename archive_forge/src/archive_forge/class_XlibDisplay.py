import ctypes
from ctypes import *
from pyglet import app
from pyglet.app.xlib import XlibSelectDevice
from .base import Display, Screen, ScreenMode, Canvas
from . import xlib_vidmoderestore
from pyglet.libs.x11 import xlib
class XlibDisplay(XlibSelectDevice, Display):
    _display = None
    _x_im = None
    _enable_xsync = False
    _screens = None

    def __init__(self, name=None, x_screen=None):
        if x_screen is None:
            x_screen = 0
        if isinstance(name, str):
            name = c_char_p(name.encode('ascii'))
        self._display = xlib.XOpenDisplay(name)
        if not self._display:
            raise NoSuchDisplayException(f'Cannot connect to "{name}"')
        screen_count = xlib.XScreenCount(self._display)
        if x_screen >= screen_count:
            raise NoSuchDisplayException(f'Display "{name}" has no screen {x_screen:d}')
        super(XlibDisplay, self).__init__()
        self.name = name
        self.x_screen = x_screen
        self._fileno = xlib.XConnectionNumber(self._display)
        self._window_map = {}
        if _have_xsync:
            event_base = c_int()
            error_base = c_int()
            if xsync.XSyncQueryExtension(self._display, byref(event_base), byref(error_base)):
                major_version = c_int()
                minor_version = c_int()
                if xsync.XSyncInitialize(self._display, byref(major_version), byref(minor_version)):
                    self._enable_xsync = True
        app.platform_event_loop.select_devices.add(self)

    def get_screens(self):
        if self._screens:
            return self._screens
        if _have_xinerama and xinerama.XineramaIsActive(self._display):
            number = c_int()
            infos = xinerama.XineramaQueryScreens(self._display, byref(number))
            infos = cast(infos, POINTER(xinerama.XineramaScreenInfo * number.value)).contents
            self._screens = []
            using_xinerama = number.value > 1
            for info in infos:
                self._screens.append(XlibScreen(self, info.x_org, info.y_org, info.width, info.height, using_xinerama))
            xlib.XFree(infos)
        else:
            screen_info = xlib.XScreenOfDisplay(self._display, self.x_screen)
            screen = XlibScreen(self, 0, 0, screen_info.contents.width, screen_info.contents.height, False)
            self._screens = [screen]
        return self._screens

    def fileno(self):
        return self._fileno

    def select(self):
        e = xlib.XEvent()
        while xlib.XPending(self._display):
            xlib.XNextEvent(self._display, e)
            if e.xany.type not in (xlib.KeyPress, xlib.KeyRelease):
                if xlib.XFilterEvent(e, e.xany.window):
                    continue
            try:
                dispatch = self._window_map[e.xany.window]
            except KeyError:
                continue
            dispatch(e)

    def poll(self):
        return xlib.XPending(self._display)