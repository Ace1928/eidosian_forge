import struct
from io import BytesIO
from functools import wraps
import warnings
def unpack_farray(self, n, unpack_item):
    list = []
    for i in range(n):
        list.append(unpack_item())
    return list