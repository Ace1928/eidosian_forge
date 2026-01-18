import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
@pytest.mark.parametrize('dt_str', ['<i2', '<i4', '<i8', '<f4', '<f8', '>i2', '>i4', '>i8', '>f4', '>f8', '|u1'])
@pytest.mark.parametrize('channels', [1, 2, 5])
@pytest.mark.parametrize('rate', [8000, 32000])
@pytest.mark.parametrize('mmap', [False, True])
@pytest.mark.parametrize('realfile', [False, True])
def test_write_roundtrip(realfile, mmap, rate, channels, dt_str, tmpdir):
    dtype = np.dtype(dt_str)
    if realfile:
        tmpfile = str(tmpdir.join('temp.wav'))
    else:
        tmpfile = BytesIO()
    data = np.random.rand(100, channels)
    if channels == 1:
        data = data[:, 0]
    if dtype.kind == 'f':
        data = data.astype(dtype)
    else:
        data = (data * 128).astype(dtype)
    wavfile.write(tmpfile, rate, data)
    rate2, data2 = wavfile.read(tmpfile, mmap=mmap)
    assert_equal(rate, rate2)
    assert_(data2.dtype.byteorder in ('<', '=', '|'), msg=data2.dtype)
    assert_array_equal(data, data2)
    if realfile:
        data2[0] = 0
    else:
        with pytest.raises(ValueError, match='read-only'):
            data2[0] = 0
    if realfile and mmap and IS_PYPY and (sys.platform == 'win32'):
        break_cycles()
        break_cycles()