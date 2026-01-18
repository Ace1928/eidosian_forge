import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def unmake_tag(i):
    b = struct.pack('>I', i)
    return b.decode('ascii', errors='replace')