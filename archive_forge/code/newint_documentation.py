from __future__ import division
import struct
from future.types.newbytes import newbytes
from future.types.newobject import newobject
from future.utils import PY3, isint, istext, isbytes, with_metaclass, native

        Return the integer represented by the given array of bytes.

        The mybytes argument must either support the buffer protocol or be an
        iterable object producing bytes.  Bytes and bytearray are examples of
        built-in objects that support the buffer protocol.

        The byteorder argument determines the byte order used to represent the
        integer.  If byteorder is 'big', the most significant byte is at the
        beginning of the byte array.  If byteorder is 'little', the most
        significant byte is at the end of the byte array.  To request the native
        byte order of the host system, use `sys.byteorder' as the byte order value.

        The signed keyword-only argument indicates whether two's complement is
        used to represent the integer.
        