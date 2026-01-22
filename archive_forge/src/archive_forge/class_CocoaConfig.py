import platform
from ctypes import c_uint32, c_int, byref
from pyglet.gl.base import Config, CanvasConfig, Context
from pyglet.gl import ContextException
from pyglet.canvas.cocoa import CocoaCanvas
from pyglet.libs.darwin import cocoapy, quartz
class CocoaConfig(Config):

    def match(self, canvas):
        attrs = []
        for name, value in self.get_gl_attributes():
            attr = _gl_attributes.get(name)
            if not attr or not value:
                continue
            attrs.append(attr)
            if attr not in _boolean_gl_attributes:
                attrs.append(int(value))
        attrs.append(cocoapy.NSOpenGLPFAAllRenderers)
        attrs.append(cocoapy.NSOpenGLPFAMaximumPolicy)
        if _os_x_version < os_x_release['snow_leopard']:
            attrs.append(cocoapy.NSOpenGLPFAFullScreen)
            attrs.append(cocoapy.NSOpenGLPFAScreenMask)
            attrs.append(quartz.CGDisplayIDToOpenGLDisplayMask(quartz.CGMainDisplayID()))
        elif _os_x_version >= os_x_release['lion']:
            version = (getattr(self, 'major_version', None) or 3, getattr(self, 'minor_version', None) or 3)
            attrs.append(cocoapy.NSOpenGLPFAOpenGLProfile)
            if version[0] >= 4 and _os_x_version >= os_x_release['mavericks']:
                attrs.append(int(cocoapy.NSOpenGLProfileVersion4_1Core))
            elif version[0] >= 3:
                attrs.append(int(cocoapy.NSOpenGLProfileVersion3_2Core))
            else:
                attrs.append(int(cocoapy.NSOpenGLProfileVersionLegacy))
        attrs.append(0)
        attrsArrayType = c_uint32 * len(attrs)
        attrsArray = attrsArrayType(*attrs)
        pixel_format = NSOpenGLPixelFormat.alloc().initWithAttributes_(attrsArray)
        if pixel_format is None:
            return []
        else:
            return [CocoaCanvasConfig(canvas, self, pixel_format)]