from collections import namedtuple
import builtins
import struct
import sys
def _byteswap(data, width):
    swapped_data = bytearray(len(data))
    for i in range(0, len(data), width):
        for j in range(width):
            swapped_data[i + width - 1 - j] = data[i + j]
    return bytes(swapped_data)