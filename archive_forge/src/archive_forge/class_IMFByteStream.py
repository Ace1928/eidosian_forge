import os
import platform
import warnings
from pyglet import image
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32 import com
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.media import Source
from pyglet.media.codecs import AudioFormat, AudioData, VideoFormat, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class IMFByteStream(com.pIUnknown):
    _methods_ = [('GetCapabilities', com.STDMETHOD()), ('GetLength', com.STDMETHOD()), ('SetLength', com.STDMETHOD()), ('GetCurrentPosition', com.STDMETHOD()), ('SetCurrentPosition', com.STDMETHOD(c_ulonglong)), ('IsEndOfStream', com.STDMETHOD()), ('Read', com.STDMETHOD()), ('BeginRead', com.STDMETHOD()), ('EndRead', com.STDMETHOD()), ('Write', com.STDMETHOD(POINTER(BYTE), ULONG, POINTER(ULONG))), ('BeginWrite', com.STDMETHOD()), ('EndWrite', com.STDMETHOD()), ('Seek', com.STDMETHOD()), ('Flush', com.STDMETHOD()), ('Close', com.STDMETHOD())]