import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage
import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray
import unittest
def _gen_alignment_data(dtype=float32, type='binary', max_size=24):
    """
    generator producing data with different alignment and offsets
    to test simd vectorization

    Parameters
    ----------
    dtype : dtype
        data type to produce
    type : string
        'unary': create data for unary operations, creates one input
                 and output array
        'binary': create data for unary operations, creates two input
                 and output array
    max_size : integer
        maximum size of data to produce

    Returns
    -------
    if type is 'unary' yields one output, one input array and a message
    containing information on the data
    if type is 'binary' yields one output array, two input array and a message
    containing information on the data

    """
    ufmt = 'unary offset=(%d, %d), size=%d, dtype=%r, %s'
    bfmt = 'binary offset=(%d, %d, %d), size=%d, dtype=%r, %s'
    for o in range(3):
        for s in range(o + 2, max(o + 3, max_size)):
            if type == 'unary':

                def inp():
                    return arange(s, dtype=dtype)[o:]
                out = empty((s,), dtype=dtype)[o:]
                yield (out, inp(), ufmt % (o, o, s, dtype, 'out of place'))
                d = inp()
                yield (d, d, ufmt % (o, o, s, dtype, 'in place'))
                yield (out[1:], inp()[:-1], ufmt % (o + 1, o, s - 1, dtype, 'out of place'))
                yield (out[:-1], inp()[1:], ufmt % (o, o + 1, s - 1, dtype, 'out of place'))
                yield (inp()[:-1], inp()[1:], ufmt % (o, o + 1, s - 1, dtype, 'aliased'))
                yield (inp()[1:], inp()[:-1], ufmt % (o + 1, o, s - 1, dtype, 'aliased'))
            if type == 'binary':

                def inp1():
                    return arange(s, dtype=dtype)[o:]
                inp2 = inp1
                out = empty((s,), dtype=dtype)[o:]
                yield (out, inp1(), inp2(), bfmt % (o, o, o, s, dtype, 'out of place'))
                d = inp1()
                yield (d, d, inp2(), bfmt % (o, o, o, s, dtype, 'in place1'))
                d = inp2()
                yield (d, inp1(), d, bfmt % (o, o, o, s, dtype, 'in place2'))
                yield (out[1:], inp1()[:-1], inp2()[:-1], bfmt % (o + 1, o, o, s - 1, dtype, 'out of place'))
                yield (out[:-1], inp1()[1:], inp2()[:-1], bfmt % (o, o + 1, o, s - 1, dtype, 'out of place'))
                yield (out[:-1], inp1()[:-1], inp2()[1:], bfmt % (o, o, o + 1, s - 1, dtype, 'out of place'))
                yield (inp1()[1:], inp1()[:-1], inp2()[:-1], bfmt % (o + 1, o, o, s - 1, dtype, 'aliased'))
                yield (inp1()[:-1], inp1()[1:], inp2()[:-1], bfmt % (o, o + 1, o, s - 1, dtype, 'aliased'))
                yield (inp1()[:-1], inp1()[:-1], inp2()[1:], bfmt % (o, o, o + 1, s - 1, dtype, 'aliased'))