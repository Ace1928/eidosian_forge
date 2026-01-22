from ctypes import *
from ctypes import _SimpleCData, _check_size
from OpenGL import extensions
from OpenGL.raw.GL._types import *
from OpenGL._bytes import as_8_bit
from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
class _WGLQuerier(extensions.ExtensionQuerier):
    prefix = b'WGL_'
    assumed_version = [1, 0]
    version_prefix = b'WGL_VERSION_WGL_'

    def pullVersion(self):
        return [1, 0]

    def pullExtensions(self):
        from OpenGL.platform import PLATFORM
        wglGetCurrentDC = PLATFORM.OpenGL.wglGetCurrentDC
        wglGetCurrentDC.restyle = HDC
        try:
            dc = wglGetCurrentDC()
            proc_address = PLATFORM.getExtensionProcedure(b'wglGetExtensionsStringARB')
            wglGetExtensionStringARB = PLATFORM.functionTypeFor(PLATFORM.WGL)(c_char_p, HDC)(proc_address)
        except TypeError as err:
            return None
        except AttributeError as err:
            return []
        else:
            return wglGetExtensionStringARB(dc).split()