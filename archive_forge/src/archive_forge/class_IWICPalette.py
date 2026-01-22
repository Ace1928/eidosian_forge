from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICPalette(com.pIUnknown):
    _methods_ = [('InitializePredefined', com.STDMETHOD()), ('InitializeCustom', com.STDMETHOD()), ('InitializeFromBitmap', com.STDMETHOD()), ('InitializeFromPalette', com.STDMETHOD()), ('GetType', com.STDMETHOD()), ('GetColorCount', com.STDMETHOD()), ('GetColors', com.STDMETHOD()), ('IsBlackWhite', com.STDMETHOD()), ('IsGrayscale', com.STDMETHOD()), ('HasAlpha', com.STDMETHOD())]