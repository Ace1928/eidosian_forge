import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _init_from_file(self, library, face, index, path):
    u_filename = c_char_p(_encode_filename(path))
    error = FT_New_Face(library, u_filename, index, byref(face))
    return error