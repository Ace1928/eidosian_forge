from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class TextElements16(TextElements8):
    string_textitem = Struct(LengthOf('string', 1), Int8('delta'), String16('string', pad=0))