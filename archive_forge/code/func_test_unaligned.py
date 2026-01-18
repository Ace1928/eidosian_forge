import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_unaligned(self):
    v = (np.zeros(64, dtype=np.int8) + ord('a'))[1:-7]
    d = v.view(np.dtype('S8'))
    x = (np.zeros(16, dtype=np.int8) + ord('a'))[1:-7]
    x = x.view(np.dtype('S8'))
    x[...] = np.array('b' * 8, dtype='S')
    b = np.arange(d.size)
    assert_equal(d[b], d)
    d[b] = x
    b = np.zeros(d.size + 1).view(np.int8)[1:-(np.intp(0).itemsize - 1)]
    b = b.view(np.intp)[:d.size]
    b[...] = np.arange(d.size)
    assert_equal(d[b.astype(np.int16)], d)
    d[b.astype(np.int16)] = x
    d[b % 2 == 0]
    d[b % 2 == 0] = x[::2]