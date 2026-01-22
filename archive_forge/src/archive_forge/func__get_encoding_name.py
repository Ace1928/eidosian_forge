import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def _get_encoding_name(self):
    encoding = self.encoding
    for key, value in FT_ENCODINGS.items():
        if encoding == value:
            return key
    return 'Unknown encoding'