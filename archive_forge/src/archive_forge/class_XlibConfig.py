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
class XlibConfig(Config):

    def match(self, canvas):
        if not isinstance(canvas, XlibCanvas):
            raise RuntimeError('Canvas must be an instance of XlibCanvas')
        x_display = canvas.display._display
        x_screen = canvas.display.x_screen
        info = glx_info.GLXInfo(x_display)
        attrs = []
        for name, value in self.get_gl_attributes():
            attr = XlibCanvasConfig.attribute_ids.get(name, None)
            if attr and value is not None:
                attrs.extend([attr, int(value)])
        attrs.extend([glx.GLX_X_RENDERABLE, True])
        attrs.extend([0, 0])
        attrib_list = (c_int * len(attrs))(*attrs)
        elements = c_int()
        configs = glx.glXChooseFBConfig(x_display, x_screen, attrib_list, byref(elements))
        if not configs:
            return []
        configs = cast(configs, POINTER(glx.GLXFBConfig * elements.value)).contents
        result = [XlibCanvasConfig(canvas, info, c, self) for c in configs]
        return result