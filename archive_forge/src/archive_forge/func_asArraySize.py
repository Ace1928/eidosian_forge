import OpenGL
import ctypes
from OpenGL import _configflags
from OpenGL import contextdata, error, converters
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,unicode
import logging
from OpenGL import acceleratesupport
def asArraySize(incoming, function, args):
    handler = typ.getHandler(incoming)
    result = handler.asArray(incoming, dataType)
    byteSize = handler.arrayByteCount(result)
    if byteSize != expectedBytes:
        raise ValueError('Expected %r byte array, got %r byte array' % (expectedBytes, byteSize), incoming)
    return result