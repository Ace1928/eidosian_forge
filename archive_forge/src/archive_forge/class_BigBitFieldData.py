from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
class BigBitFieldData(object):

    def __init__(self, instance, name):
        self.instance = instance
        self.name = name
        value = self.instance.__data__.get(self.name)
        if not value:
            value = bytearray()
        elif not isinstance(value, bytearray):
            value = bytearray(value)
        self._buffer = self.instance.__data__[self.name] = value

    def clear(self):
        self._buffer.clear()

    def _ensure_length(self, idx):
        byte_num, byte_offset = divmod(idx, 8)
        cur_size = len(self._buffer)
        if cur_size <= byte_num:
            self._buffer.extend(b'\x00' * (byte_num + 1 - cur_size))
        return (byte_num, byte_offset)

    def set_bit(self, idx):
        byte_num, byte_offset = self._ensure_length(idx)
        self._buffer[byte_num] |= 1 << byte_offset

    def clear_bit(self, idx):
        byte_num, byte_offset = self._ensure_length(idx)
        self._buffer[byte_num] &= ~(1 << byte_offset)

    def toggle_bit(self, idx):
        byte_num, byte_offset = self._ensure_length(idx)
        self._buffer[byte_num] ^= 1 << byte_offset
        return bool(self._buffer[byte_num] & 1 << byte_offset)

    def is_set(self, idx):
        byte_num, byte_offset = divmod(idx, 8)
        cur_size = len(self._buffer)
        if cur_size <= byte_num:
            return False
        return bool(self._buffer[byte_num] & 1 << byte_offset)
    __getitem__ = is_set

    def __setitem__(self, item, value):
        self.set_bit(item) if value else self.clear_bit(item)
    __delitem__ = clear_bit

    def __len__(self):
        return len(self._buffer)

    def _get_compatible_data(self, other):
        if isinstance(other, BigBitFieldData):
            data = other._buffer
        elif isinstance(other, (bytes, bytearray, memoryview)):
            data = other
        else:
            raise ValueError('Incompatible data-type')
        diff = len(data) - len(self)
        if diff > 0:
            self._buffer.extend(b'\x00' * diff)
        return data

    def _bitwise_op(self, other, op):
        if isinstance(other, BigBitFieldData):
            data = other._buffer
        elif isinstance(other, (bytes, bytearray, memoryview)):
            data = other
        else:
            raise ValueError('Incompatible data-type')
        buf = bytearray(b'\x00' * max(len(self), len(other)))
        it = itertools.zip_longest(self._buffer, data, fillvalue=0)
        for i, (a, b) in enumerate(it):
            buf[i] = op(a, b)
        return buf

    def __and__(self, other):
        return self._bitwise_op(other, operator.and_)

    def __or__(self, other):
        return self._bitwise_op(other, operator.or_)

    def __xor__(self, other):
        return self._bitwise_op(other, operator.xor)

    def __iter__(self):
        for b in self._buffer:
            for j in range(8):
                yield (1 if b & 1 << j else 0)

    def __repr__(self):
        return repr(self._buffer)
    if sys.version_info[0] < 3:

        def __str__(self):
            return bytes_type(self._buffer)
    else:

        def __bytes__(self):
            return bytes_type(self._buffer)