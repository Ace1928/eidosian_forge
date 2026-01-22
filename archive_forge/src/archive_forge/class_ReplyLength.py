from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class ReplyLength(TotalLengthField):
    structcode = 'L'
    structvalues = 1

    def calc_length(self, length):
        return (length - 32) // 4