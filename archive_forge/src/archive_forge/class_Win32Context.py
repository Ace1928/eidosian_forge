from pyglet.canvas.win32 import Win32Canvas
from .base import Config, CanvasConfig, Context
from pyglet import gl
from pyglet.gl import gl_info
from pyglet.gl import wgl
from pyglet.gl import wglext_arb
from pyglet.gl import wgl_info
from pyglet.libs.win32 import _user32, _kernel32, _gdi32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class Win32Context(_BaseWin32Context):

    def attach(self, canvas):
        super().attach(canvas)
        if not self._context:
            self.config._set_pixel_format(canvas)
            self._context = wgl.wglCreateContext(canvas.hdc)
        share = self.context_share
        if share:
            if not share.canvas:
                raise RuntimeError('Share context has no canvas.')
            if not wgl.wglShareLists(share._context, self._context):
                raise gl.ContextException('Unable to share contexts.')