from ctypes import *
from pyglet import gl
from pyglet.canvas.headless import HeadlessCanvas
from pyglet.libs.egl import egl
from pyglet.libs.egl.egl import *
from .base import CanvasConfig, Config, Context
class HeadlessConfig(Config):

    def match(self, canvas):
        if not isinstance(canvas, HeadlessCanvas):
            raise RuntimeError('Canvas must be an instance of HeadlessCanvas')
        display_connection = canvas.display._display_connection
        attrs = []
        for name, value in self.get_gl_attributes():
            if name == 'double_buffer':
                continue
            attr = HeadlessCanvasConfig.attribute_ids.get(name, None)
            if attr and value is not None:
                attrs.extend([attr, int(value)])
        attrs.extend([EGL_SURFACE_TYPE, EGL_PBUFFER_BIT])
        if self.opengl_api == 'gl':
            attrs.extend([EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT])
        elif self.opengl_api == 'gles':
            attrs.extend([EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT])
        else:
            raise ValueError(f'Unknown OpenGL API: {self.opengl_api}')
        attrs.extend([EGL_NONE])
        attrs_list = (egl.EGLint * len(attrs))(*attrs)
        num_config = egl.EGLint()
        egl.eglChooseConfig(display_connection, attrs_list, None, 0, byref(num_config))
        configs = (egl.EGLConfig * num_config.value)()
        egl.eglChooseConfig(display_connection, attrs_list, configs, num_config.value, byref(num_config))
        result = [HeadlessCanvasConfig(canvas, c, self) for c in configs]
        return result