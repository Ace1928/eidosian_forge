import struct
from io import BytesIO
from functools import wraps
import warnings
def unpack_uint(self):
    i = self.__pos
    self.__pos = j = i + 4
    data = self.__buf[i:j]
    if len(data) < 4:
        raise EOFError
    return struct.unpack('>L', data)[0]