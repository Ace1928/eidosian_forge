import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_points(self):
    n = self._FT_Outline.n_points
    data = []
    for i in range(n):
        v = self._FT_Outline.points[i]
        data.append((v.x, v.y))
    return data