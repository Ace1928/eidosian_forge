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
class Win32CanvasConfigARB(CanvasConfig):
    attribute_ids = {'double_buffer': wglext_arb.WGL_DOUBLE_BUFFER_ARB, 'stereo': wglext_arb.WGL_STEREO_ARB, 'buffer_size': wglext_arb.WGL_COLOR_BITS_ARB, 'aux_buffers': wglext_arb.WGL_AUX_BUFFERS_ARB, 'sample_buffers': wglext_arb.WGL_SAMPLE_BUFFERS_ARB, 'samples': wglext_arb.WGL_SAMPLES_ARB, 'red_size': wglext_arb.WGL_RED_BITS_ARB, 'green_size': wglext_arb.WGL_GREEN_BITS_ARB, 'blue_size': wglext_arb.WGL_BLUE_BITS_ARB, 'alpha_size': wglext_arb.WGL_ALPHA_BITS_ARB, 'depth_size': wglext_arb.WGL_DEPTH_BITS_ARB, 'stencil_size': wglext_arb.WGL_STENCIL_BITS_ARB, 'accum_red_size': wglext_arb.WGL_ACCUM_RED_BITS_ARB, 'accum_green_size': wglext_arb.WGL_ACCUM_GREEN_BITS_ARB, 'accum_blue_size': wglext_arb.WGL_ACCUM_BLUE_BITS_ARB, 'accum_alpha_size': wglext_arb.WGL_ACCUM_ALPHA_BITS_ARB}

    def __init__(self, canvas, pf, config):
        super().__init__(canvas, config)
        self._pf = pf
        names = list(self.attribute_ids.keys())
        attrs = list(self.attribute_ids.values())
        attrs = (c_int * len(attrs))(*attrs)
        values = (c_int * len(attrs))()
        wglext_arb.wglGetPixelFormatAttribivARB(canvas.hdc, pf, 0, len(attrs), attrs, values)
        for name, value in zip(names, values):
            setattr(self, name, value)

    def compatible(self, canvas):
        return isinstance(canvas, Win32Canvas)

    def create_context(self, share):
        if wgl_info.have_extension('WGL_ARB_create_context'):
            return Win32ARBContext(self, share)
        else:
            return Win32Context(self, share)

    def _set_pixel_format(self, canvas):
        _gdi32.SetPixelFormat(canvas.hdc, self._pf, None)