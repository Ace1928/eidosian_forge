import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_contours(self):
    n = self._FT_Outline.n_contours
    data = [self._FT_Outline.contours[i] for i in range(n)]
    return data