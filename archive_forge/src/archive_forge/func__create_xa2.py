from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def _create_xa2(self, device_id=None):
    self._xaudio2 = lib.IXAudio2()
    try:
        lib.XAudio2Create(ctypes.byref(self._xaudio2), 0, self.processor)
    except OSError:
        raise ImportError('XAudio2 driver could not be initialized.')
    if _debug:
        debug = lib.XAUDIO2_DEBUG_CONFIGURATION()
        debug.LogThreadID = True
        debug.TraceMask = lib.XAUDIO2_LOG_ERRORS | lib.XAUDIO2_LOG_WARNINGS
        debug.BreakMask = lib.XAUDIO2_LOG_WARNINGS
        self._xaudio2.SetDebugConfiguration(ctypes.byref(debug), None)
    self._xaudio2.RegisterForCallbacks(self._engine_callback)
    self._mvoice_details = lib.XAUDIO2_VOICE_DETAILS()
    self._master_voice = lib.IXAudio2MasteringVoice()
    self._xaudio2.CreateMasteringVoice(byref(self._master_voice), lib.XAUDIO2_DEFAULT_CHANNELS, lib.XAUDIO2_DEFAULT_SAMPLERATE, 0, device_id, None, self.category)
    self._master_voice.GetVoiceDetails(byref(self._mvoice_details))
    self._x3d_handle = None
    self._dsp_settings = None
    if self.allow_3d:
        self.enable_3d()