from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICBitmap(IWICBitmapSource, com.pIUnknown):
    _methods_ = [('Lock', com.STDMETHOD()), ('SetPalette', com.STDMETHOD()), ('SetResolution', com.STDMETHOD())]