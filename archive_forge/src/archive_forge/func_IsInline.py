import array
import contextlib
import enum
import struct
@staticmethod
def IsInline(type_):
    return type_ <= Type.FLOAT or type_ == Type.BOOL