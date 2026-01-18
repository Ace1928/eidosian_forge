import struct
from io import BytesIO
from functools import wraps
import warnings
def unpack_list(self, unpack_item):
    list = []
    while 1:
        x = self.unpack_uint()
        if x == 0:
            break
        if x != 1:
            raise ConversionError('0 or 1 expected, got %r' % (x,))
        item = unpack_item()
        list.append(item)
    return list