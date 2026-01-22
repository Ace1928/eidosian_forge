from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICBitmapFrameEncode(com.pIUnknown):
    _methods_ = [('Initialize', com.STDMETHOD(IPropertyBag2)), ('SetSize', com.STDMETHOD(UINT, UINT)), ('SetResolution', com.STDMETHOD()), ('SetPixelFormat', com.STDMETHOD(REFWICPixelFormatGUID)), ('SetColorContexts', com.STDMETHOD()), ('SetPalette', com.STDMETHOD(IWICPalette)), ('SetThumbnail', com.STDMETHOD()), ('WritePixels', com.STDMETHOD(UINT, UINT, UINT, POINTER(BYTE))), ('WriteSource', com.STDMETHOD()), ('Commit', com.STDMETHOD()), ('GetMetadataQueryWriter', com.STDMETHOD())]