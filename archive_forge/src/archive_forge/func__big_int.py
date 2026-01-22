from __future__ import absolute_import, division, print_function
import os
import re
from contextlib import contextmanager
from struct import Struct
from ansible.module_utils.six import PY3
@classmethod
def _big_int(cls, raw_string, byte_order, signed=False):
    if byte_order not in ('big', 'little'):
        raise ValueError('Byte_order must be one of (big, little) not %s' % byte_order)
    if PY3:
        return int.from_bytes(raw_string, byte_order, signed=signed)
    result = 0
    byte_length = len(raw_string)
    if byte_length > 0:
        msb = raw_string[0] if byte_order == 'big' else raw_string[-1]
        negative = bool(ord(msb) & 128)
        pad = b'\xff' if signed and negative else b'\x00'
        pad_length = 4 - byte_length % 4
        if pad_length < 4:
            raw_string = pad * pad_length + raw_string if byte_order == 'big' else raw_string + pad * pad_length
            byte_length += pad_length
        if byte_order == 'big':
            for i in range(0, byte_length, cls.UINT32_OFFSET):
                left_shift = result << cls.UINT32_OFFSET * 8
                result = left_shift + _UINT32.unpack(raw_string[i:i + cls.UINT32_OFFSET])[0]
        else:
            for i in range(byte_length, 0, -cls.UINT32_OFFSET):
                left_shift = result << cls.UINT32_OFFSET * 8
                result = left_shift + _UINT32_LE.unpack(raw_string[i - cls.UINT32_OFFSET:i])[0]
        if signed and negative:
            result -= 1 << 8 * byte_length
    return result