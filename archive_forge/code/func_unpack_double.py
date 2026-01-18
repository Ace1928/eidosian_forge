import struct
from io import BytesIO
from functools import wraps
import warnings
def unpack_double(self):
    i = self.__pos
    self.__pos = j = i + 8
    data = self.__buf[i:j]
    if len(data) < 8:
        raise EOFError
    return struct.unpack('>d', data)[0]