import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_postscript_name(self):
    return FT_Get_Postscript_Name(self._FT_Face)