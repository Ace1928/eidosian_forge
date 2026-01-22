import ctypes
from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
from OpenGL import platform as _p
from OpenGL import extensions 
from OpenGL._bytes import as_8_bit
class _EGLQuerier(extensions.ExtensionQuerier):
    prefix = as_8_bit('EGL_')
    assumed_version = [1, 0]
    version_prefix = as_8_bit('EGL_VERSION_EGL_')

    def getDisplay(self):
        """Retrieve the currently-bound, or the default, display"""
        from OpenGL.EGL import eglGetCurrentDisplay, eglGetDisplay, EGL_DEFAULT_DISPLAY
        return eglGetCurrentDisplay() or eglGetDisplay(EGL_DEFAULT_DISPLAY)

    def pullVersion(self):
        from OpenGL.EGL import eglQueryString, EGL_VERSION
        return eglQueryString(self.getDisplay(), EGL_VERSION)

    def pullExtensions(self):
        from OpenGL.EGL import eglQueryString, EGL_EXTENSIONS
        return eglQueryString(self.getDisplay(), EGL_EXTENSIONS)