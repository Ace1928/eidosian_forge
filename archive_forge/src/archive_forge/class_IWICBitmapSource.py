from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICBitmapSource(com.pIUnknown):
    _methods_ = [('GetSize', com.STDMETHOD(POINTER(UINT), POINTER(UINT))), ('GetPixelFormat', com.STDMETHOD(REFWICPixelFormatGUID)), ('GetResolution', com.STDMETHOD(POINTER(DOUBLE), POINTER(DOUBLE))), ('CopyPalette', com.STDMETHOD()), ('CopyPixels', com.STDMETHOD(c_void_p, UINT, UINT, c_void_p))]