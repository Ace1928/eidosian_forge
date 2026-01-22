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
class Win32CanvasConfig(CanvasConfig):

    def __init__(self, canvas, pf, config):
        super().__init__(canvas, config)
        self._pf = pf
        self._pfd = PIXELFORMATDESCRIPTOR()
        _gdi32.DescribePixelFormat(canvas.hdc, pf, sizeof(PIXELFORMATDESCRIPTOR), byref(self._pfd))
        self.double_buffer = bool(self._pfd.dwFlags & PFD_DOUBLEBUFFER)
        self.sample_buffers = 0
        self.samples = 0
        self.stereo = bool(self._pfd.dwFlags & PFD_STEREO)
        self.buffer_size = self._pfd.cColorBits
        self.red_size = self._pfd.cRedBits
        self.green_size = self._pfd.cGreenBits
        self.blue_size = self._pfd.cBlueBits
        self.alpha_size = self._pfd.cAlphaBits
        self.accum_red_size = self._pfd.cAccumRedBits
        self.accum_green_size = self._pfd.cAccumGreenBits
        self.accum_blue_size = self._pfd.cAccumBlueBits
        self.accum_alpha_size = self._pfd.cAccumAlphaBits
        self.depth_size = self._pfd.cDepthBits
        self.stencil_size = self._pfd.cStencilBits
        self.aux_buffers = self._pfd.cAuxBuffers

    def compatible(self, canvas):
        return isinstance(canvas, Win32Canvas)

    def create_context(self, share):
        return Win32Context(self, share)

    def _set_pixel_format(self, canvas):
        _gdi32.SetPixelFormat(canvas.hdc, self._pf, byref(self._pfd))