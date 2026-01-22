import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class XAUDIO2_VOICE_STATE(ctypes.Structure):
    _fields_ = [('pCurrentBufferContext', c_void_p), ('BuffersQueued', UINT32), ('SamplesPlayed', UINT32)]

    def __repr__(self):
        return 'XAUDIO2_VOICE_STATE(BuffersQueued={0}, SamplesPlayed={1})'.format(self.BuffersQueued, self.SamplesPlayed)