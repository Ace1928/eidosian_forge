from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICBitmapEncoder(com.pIUnknown):
    _methods_ = [('Initialize', com.STDMETHOD(IWICStream, WICBitmapEncoderCacheOption)), ('GetContainerFormat', com.STDMETHOD()), ('GetEncoderInfo', com.STDMETHOD()), ('SetColorContexts', com.STDMETHOD()), ('SetPalette', com.STDMETHOD()), ('SetThumbnail', com.STDMETHOD()), ('SetPreview', com.STDMETHOD()), ('CreateNewFrame', com.STDMETHOD(POINTER(IWICBitmapFrameEncode), POINTER(IPropertyBag2))), ('Commit', com.STDMETHOD()), ('GetMetadataQueryWriter', com.STDMETHOD())]