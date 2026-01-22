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
def _get_arb_pixel_format_matching_configs(self, canvas):
    """Get configs using the WGL_ARB_pixel_format extension.
        This method assumes a (dummy) GL context is already created."""
    if self.sample_buffers or self.samples:
        if not gl_info.have_extension('GL_ARB_multisample'):
            return []
    attrs = []
    for name, value in self.get_gl_attributes():
        attr = Win32CanvasConfigARB.attribute_ids.get(name, None)
        if attr and value is not None:
            attrs.extend([attr, int(value)])
    attrs.append(0)
    attrs = (c_int * len(attrs))(*attrs)
    pformats = (c_int * 16)()
    nformats = c_uint(16)
    wglext_arb.wglChoosePixelFormatARB(canvas.hdc, attrs, None, nformats, pformats, nformats)
    formats = [Win32CanvasConfigARB(canvas, pf, self) for pf in pformats[:nformats.value]]
    return formats