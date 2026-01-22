import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class IXAudio2SourceVoice(IXAudio2Voice):
    _methods_ = [('Start', com.STDMETHOD(UINT32, UINT32)), ('Stop', com.STDMETHOD(UINT32, UINT32)), ('SubmitSourceBuffer', com.STDMETHOD(POINTER(XAUDIO2_BUFFER), c_void_p)), ('FlushSourceBuffers', com.STDMETHOD()), ('Discontinuity', com.STDMETHOD()), ('ExitLoop', com.STDMETHOD(UINT32)), ('GetState', com.VOIDMETHOD(POINTER(XAUDIO2_VOICE_STATE), UINT32)), ('SetFrequencyRatio', com.STDMETHOD(c_float, UINT32)), ('GetFrequencyRatio', com.VOIDMETHOD(POINTER(c_float))), ('SetSourceSampleRate', com.STDMETHOD(UINT32))]