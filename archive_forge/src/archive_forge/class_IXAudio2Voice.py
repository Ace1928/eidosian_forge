import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class IXAudio2Voice(com.pInterface):
    _methods_ = [('GetVoiceDetails', com.VOIDMETHOD(POINTER(XAUDIO2_VOICE_DETAILS))), ('SetOutputVoices', com.STDMETHOD(POINTER(XAUDIO2_VOICE_SENDS))), ('SetEffectChain', com.STDMETHOD(POINTER(XAUDIO2_EFFECT_CHAIN))), ('EnableEffect', com.STDMETHOD(UINT32, UINT32)), ('DisableEffect', com.STDMETHOD(UINT32, UINT32)), ('GetEffectState', com.VOIDMETHOD(UINT32, POINTER(BOOL))), ('SetEffectParameters', com.STDMETHOD(UINT32, c_void_p, UINT32, UINT32)), ('GetEffectParameters', com.VOIDMETHOD(UINT32, POINTER(BOOL))), ('SetFilterParameters', com.STDMETHOD(POINTER(XAUDIO2_FILTER_PARAMETERS), UINT32)), ('GetFilterParameters', com.VOIDMETHOD(POINTER(XAUDIO2_FILTER_PARAMETERS))), ('SetOutputFilterParameters', com.STDMETHOD(c_void_p, POINTER(XAUDIO2_FILTER_PARAMETERS), UINT32)), ('GetOutputFilterParameters', com.VOIDMETHOD(c_void_p, POINTER(XAUDIO2_FILTER_PARAMETERS))), ('SetVolume', com.STDMETHOD(c_float, UINT32)), ('GetVolume', com.VOIDMETHOD(POINTER(c_float))), ('SetChannelVolumes', com.STDMETHOD(UINT32, POINTER(c_float), UINT32)), ('GetChannelVolumes', com.VOIDMETHOD(UINT32, POINTER(c_float))), ('SetOutputMatrix', com.STDMETHOD(c_void_p, UINT32, UINT32, POINTER(c_float), UINT32)), ('GetOutputMatrix', com.STDMETHOD(c_void_p, UINT32, UINT32, POINTER(c_float))), ('DestroyVoice', com.VOIDMETHOD())]