import warnings
from ctypes import *
from .base import Config, CanvasConfig, Context
from pyglet.canvas.xlib import XlibCanvas
from pyglet.gl import glx
from pyglet.gl import glxext_arb
from pyglet.gl import glx_info
from pyglet.gl import glxext_mesa
from pyglet.gl import lib
from pyglet import gl
def _wait_vsync(self):
    if self._vsync and self._have_SGI_video_sync and self._use_video_sync:
        count = c_uint()
        glxext_arb.glXGetVideoSyncSGI(byref(count))
        glxext_arb.glXWaitVideoSyncSGI(2, (count.value + 1) % 2, byref(count))