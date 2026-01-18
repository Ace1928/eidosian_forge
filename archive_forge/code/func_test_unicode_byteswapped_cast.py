import pytest
import textwrap
import enum
import random
import ctypes
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('order1', ['>', '<'])
@pytest.mark.parametrize('order2', ['>', '<'])
def test_unicode_byteswapped_cast(self, order1, order2):
    dtype1 = np.dtype(f'{order1}U30')
    dtype2 = np.dtype(f'{order2}U30')
    data1 = np.empty(30 * 4 + 1, dtype=np.uint8)[1:].view(dtype1)
    data2 = np.empty(30 * 4 + 1, dtype=np.uint8)[1:].view(dtype2)
    if dtype1.alignment != 1:
        assert not data1.flags.aligned
        assert not data2.flags.aligned
    element = 'this is a ünicode string‽'
    data1[()] = element
    for data in [data1, data1.copy()]:
        data2[...] = data1
        assert data2[()] == element
        assert data2.copy()[()] == element