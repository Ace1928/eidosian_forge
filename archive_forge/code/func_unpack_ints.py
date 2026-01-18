from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
@_replace_by('_tifffile.unpack_ints')
def unpack_ints(data, dtype, itemsize, runlen=0):
    """Decompress byte string to array of integers of any bit size <= 32.

    This Python implementation is slow and only handles itemsizes 1, 2, 4, 8,
    16, 32, and 64.

    Parameters
    ----------
    data : byte str
        Data to decompress.
    dtype : numpy.dtype or str
        A numpy boolean or integer type.
    itemsize : int
        Number of bits per integer.
    runlen : int
        Number of consecutive integers, after which to start at next byte.

    Examples
    --------
    >>> unpack_ints(b'a', 'B', 1)
    array([0, 1, 1, 0, 0, 0, 0, 1], dtype=uint8)
    >>> unpack_ints(b'ab', 'B', 2)
    array([1, 2, 0, 1, 1, 2, 0, 2], dtype=uint8)

    """
    if itemsize == 1:
        data = numpy.frombuffer(data, '|B')
        data = numpy.unpackbits(data)
        if runlen % 8:
            data = data.reshape(-1, runlen + (8 - runlen % 8))
            data = data[:, :runlen].reshape(-1)
        return data.astype(dtype)
    dtype = numpy.dtype(dtype)
    if itemsize in (8, 16, 32, 64):
        return numpy.frombuffer(data, dtype)
    if itemsize not in (1, 2, 4, 8, 16, 32):
        raise ValueError('itemsize not supported: %i' % itemsize)
    if dtype.kind not in 'biu':
        raise ValueError('invalid dtype')
    itembytes = next((i for i in (1, 2, 4, 8) if 8 * i >= itemsize))
    if itembytes != dtype.itemsize:
        raise ValueError('dtype.itemsize too small')
    if runlen == 0:
        runlen = 8 * len(data) // itemsize
    skipbits = runlen * itemsize % 8
    if skipbits:
        skipbits = 8 - skipbits
    shrbits = itembytes * 8 - itemsize
    bitmask = int(itemsize * '1' + '0' * shrbits, 2)
    dtypestr = '>' + dtype.char
    unpack = struct.unpack
    size = runlen * (len(data) * 8 // (runlen * itemsize + skipbits))
    result = numpy.empty((size,), dtype)
    bitcount = 0
    for i in range(size):
        start = bitcount // 8
        s = data[start:start + itembytes]
        try:
            code = unpack(dtypestr, s)[0]
        except Exception:
            code = unpack(dtypestr, s + b'\x00' * (itembytes - len(s)))[0]
        code <<= bitcount % 8
        code &= bitmask
        result[i] = code >> shrbits
        bitcount += itemsize
        if (i + 1) % runlen == 0:
            bitcount += skipbits
    return result