from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def _apply3d(self, source_voice, commit):
    """Calculates and sets output matrix and frequency ratio on the voice based on the listener and the voice's
           emitter. Commit determines the operation set, whether the settings are applied immediately (0) or to
           be committed together at a later time.
        """
    lib.X3DAudioCalculate(self._x3d_handle, self._listener.listener, source_voice._emitter, lib.default_dsp_calculation, self._dsp_settings)
    source_voice._voice.SetOutputMatrix(self._master_voice, 1, self._mvoice_details.InputChannels, self._dsp_settings.pMatrixCoefficients, commit)
    source_voice._voice.SetFrequencyRatio(self._dsp_settings.DopplerFactor, commit)