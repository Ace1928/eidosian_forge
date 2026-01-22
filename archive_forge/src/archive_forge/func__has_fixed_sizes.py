import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _has_fixed_sizes(self):
    return bool(self.face_flags & FT_FACE_FLAG_FIXED_SIZES)