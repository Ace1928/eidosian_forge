import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class DWM_BLURBEHIND(Structure):
    _fields_ = [('dwFlags', DWORD), ('fEnable', BOOL), ('hRgnBlur', HRGN), ('fTransitionOnMaximized', DWORD)]