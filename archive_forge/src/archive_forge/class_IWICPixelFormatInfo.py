from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICPixelFormatInfo(IWICComponentInfo, com.pIUnknown):
    _methods_ = [('GetFormatGUID', com.STDMETHOD(POINTER(com.GUID))), ('GetColorContext', com.STDMETHOD()), ('GetBitsPerPixel', com.STDMETHOD(POINTER(UINT))), ('GetChannelCount', com.STDMETHOD(POINTER(UINT))), ('GetChannelMask', com.STDMETHOD())]