from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def unpackBase128(data):
    """Read one to five bytes from UIntBase128-encoded input string, and return
    a tuple containing the decoded integer plus any leftover data.

    >>> unpackBase128(b'\\x3f\\x00\\x00') == (63, b"\\x00\\x00")
    True
    >>> unpackBase128(b'\\x8f\\xff\\xff\\xff\\x7f')[0] == 4294967295
    True
    >>> unpackBase128(b'\\x80\\x80\\x3f')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    TTLibError: UIntBase128 value must not start with leading zeros
    >>> unpackBase128(b'\\x8f\\xff\\xff\\xff\\xff\\x7f')[0]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    TTLibError: UIntBase128-encoded sequence is longer than 5 bytes
    >>> unpackBase128(b'\\x90\\x80\\x80\\x80\\x00')[0]  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      File "<stdin>", line 1, in ?
    TTLibError: UIntBase128 value exceeds 2**32-1
    """
    if len(data) == 0:
        raise TTLibError('not enough data to unpack UIntBase128')
    result = 0
    if byteord(data[0]) == 128:
        raise TTLibError('UIntBase128 value must not start with leading zeros')
    for i in range(woff2Base128MaxSize):
        if len(data) == 0:
            raise TTLibError('not enough data to unpack UIntBase128')
        code = byteord(data[0])
        data = data[1:]
        if result & 4261412864:
            raise TTLibError('UIntBase128 value exceeds 2**32-1')
        result = result << 7 | code & 127
        if code & 128 == 0:
            return (result, data)
    raise TTLibError('UIntBase128-encoded sequence is longer than 5 bytes')