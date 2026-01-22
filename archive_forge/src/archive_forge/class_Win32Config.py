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
class Win32Config(Config):

    def match(self, canvas):
        if not isinstance(canvas, Win32Canvas):
            raise RuntimeError('Canvas must be instance of Win32Canvas')
        if gl_info.have_context() and wgl_info.have_extension('WGL_ARB_pixel_format'):
            return self._get_arb_pixel_format_matching_configs(canvas)
        else:
            return self._get_pixel_format_descriptor_matching_configs(canvas)

    def _get_pixel_format_descriptor_matching_configs(self, canvas):
        """Get matching configs using standard PIXELFORMATDESCRIPTOR
        technique."""
        pfd = PIXELFORMATDESCRIPTOR()
        pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR)
        pfd.nVersion = 1
        pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL
        if self.double_buffer:
            pfd.dwFlags |= PFD_DOUBLEBUFFER
        else:
            pfd.dwFlags |= PFD_DOUBLEBUFFER_DONTCARE
        if self.stereo:
            pfd.dwFlags |= PFD_STEREO
        else:
            pfd.dwFlags |= PFD_STEREO_DONTCARE
        if not self.depth_size:
            pfd.dwFlags |= PFD_DEPTH_DONTCARE
        pfd.iPixelType = PFD_TYPE_RGBA
        pfd.cColorBits = self.buffer_size or 0
        pfd.cRedBits = self.red_size or 0
        pfd.cGreenBits = self.green_size or 0
        pfd.cBlueBits = self.blue_size or 0
        pfd.cAlphaBits = self.alpha_size or 0
        pfd.cAccumRedBits = self.accum_red_size or 0
        pfd.cAccumGreenBits = self.accum_green_size or 0
        pfd.cAccumBlueBits = self.accum_blue_size or 0
        pfd.cAccumAlphaBits = self.accum_alpha_size or 0
        pfd.cDepthBits = self.depth_size or 0
        pfd.cStencilBits = self.stencil_size or 0
        pfd.cAuxBuffers = self.aux_buffers or 0
        pf = _gdi32.ChoosePixelFormat(canvas.hdc, byref(pfd))
        if pf:
            return [Win32CanvasConfig(canvas, pf, self)]
        else:
            return []

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