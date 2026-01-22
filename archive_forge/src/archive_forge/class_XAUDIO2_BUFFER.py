import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class XAUDIO2_BUFFER(ctypes.Structure):
    _fields_ = [('Flags', UINT32), ('AudioBytes', UINT32), ('pAudioData', POINTER(c_char)), ('PlayBegin', UINT32), ('PlayLength', UINT32), ('LoopBegin', UINT32), ('LoopLength', UINT32), ('LoopCount', UINT32), ('pContext', c_void_p)]