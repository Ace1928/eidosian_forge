import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_outline(self):
    return Outline(self._FT_GlyphSlot.contents.outline)