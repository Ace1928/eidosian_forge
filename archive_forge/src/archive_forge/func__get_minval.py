import math
import struct
from ctypes import create_string_buffer
def _get_minval(size, signed=True):
    if not signed:
        return 0
    elif size == 1:
        return -128
    elif size == 2:
        return -32768
    elif size == 4:
        return -2147483648