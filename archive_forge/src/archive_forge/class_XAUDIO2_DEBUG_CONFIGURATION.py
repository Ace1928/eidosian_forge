import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class XAUDIO2_DEBUG_CONFIGURATION(ctypes.Structure):
    _fields_ = [('TraceMask', UINT32), ('BreakMask', UINT32), ('LogThreadID', BOOL), ('LogFileline', BOOL), ('LogFunctionName', BOOL), ('LogTiming', BOOL)]