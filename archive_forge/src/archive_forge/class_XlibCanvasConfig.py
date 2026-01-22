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
class XlibCanvasConfig(CanvasConfig):
    attribute_ids = {'buffer_size': glx.GLX_BUFFER_SIZE, 'level': glx.GLX_LEVEL, 'double_buffer': glx.GLX_DOUBLEBUFFER, 'stereo': glx.GLX_STEREO, 'aux_buffers': glx.GLX_AUX_BUFFERS, 'red_size': glx.GLX_RED_SIZE, 'green_size': glx.GLX_GREEN_SIZE, 'blue_size': glx.GLX_BLUE_SIZE, 'alpha_size': glx.GLX_ALPHA_SIZE, 'depth_size': glx.GLX_DEPTH_SIZE, 'stencil_size': glx.GLX_STENCIL_SIZE, 'accum_red_size': glx.GLX_ACCUM_RED_SIZE, 'accum_green_size': glx.GLX_ACCUM_GREEN_SIZE, 'accum_blue_size': glx.GLX_ACCUM_BLUE_SIZE, 'accum_alpha_size': glx.GLX_ACCUM_ALPHA_SIZE, 'sample_buffers': glx.GLX_SAMPLE_BUFFERS, 'samples': glx.GLX_SAMPLES, 'render_type': glx.GLX_RENDER_TYPE, 'config_caveat': glx.GLX_CONFIG_CAVEAT, 'transparent_type': glx.GLX_TRANSPARENT_TYPE, 'transparent_index_value': glx.GLX_TRANSPARENT_INDEX_VALUE, 'transparent_red_value': glx.GLX_TRANSPARENT_RED_VALUE, 'transparent_green_value': glx.GLX_TRANSPARENT_GREEN_VALUE, 'transparent_blue_value': glx.GLX_TRANSPARENT_BLUE_VALUE, 'transparent_alpha_value': glx.GLX_TRANSPARENT_ALPHA_VALUE, 'x_renderable': glx.GLX_X_RENDERABLE}

    def __init__(self, canvas, info, fbconfig, config):
        super().__init__(canvas, config)
        self.glx_info = info
        self.fbconfig = fbconfig
        for name, attr in self.attribute_ids.items():
            value = c_int()
            result = glx.glXGetFBConfigAttrib(canvas.display._display, self.fbconfig, attr, byref(value))
            if result >= 0:
                setattr(self, name, value.value)

    def get_visual_info(self):
        return glx.glXGetVisualFromFBConfig(self.canvas.display._display, self.fbconfig).contents

    def create_context(self, share):
        return XlibContext(self, share)

    def compatible(self, canvas):
        return isinstance(canvas, XlibCanvas)

    def _create_glx_context(self, share):
        raise NotImplementedError('abstract')

    def is_complete(self):
        return True