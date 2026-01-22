import array
import contextlib
import enum
import struct
@staticmethod
def IsFixedTypedVectorElementType(type_):
    return Type.INT <= type_ <= Type.FLOAT