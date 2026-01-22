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
class IMFSample(IMFAttributes, com.pIUnknown):
    _methods_ = [('GetSampleFlags', com.STDMETHOD()), ('SetSampleFlags', com.STDMETHOD()), ('GetSampleTime', com.STDMETHOD()), ('SetSampleTime', com.STDMETHOD()), ('GetSampleDuration', com.STDMETHOD(POINTER(c_ulonglong))), ('SetSampleDuration', com.STDMETHOD(DWORD, IMFMediaBuffer)), ('GetBufferCount', com.STDMETHOD(POINTER(DWORD))), ('GetBufferByIndex', com.STDMETHOD(DWORD, IMFMediaBuffer)), ('ConvertToContiguousBuffer', com.STDMETHOD(POINTER(IMFMediaBuffer))), ('AddBuffer', com.STDMETHOD(POINTER(DWORD))), ('RemoveBufferByIndex', com.STDMETHOD()), ('RemoveAllBuffers', com.STDMETHOD()), ('GetTotalLength', com.STDMETHOD(POINTER(DWORD))), ('CopyToBuffer', com.STDMETHOD())]