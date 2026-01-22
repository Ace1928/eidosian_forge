import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class DEV_BROADCAST_HDR(Structure):
    _fields_ = (('dbch_size', DWORD), ('dbch_devicetype', DWORD), ('dbch_reserved', DWORD))