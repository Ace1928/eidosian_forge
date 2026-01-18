import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_read_5():
    for mmap in [False, True]:
        filename = 'test-44100Hz-2ch-32bit-float-be.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 44100)
        assert_(np.issubdtype(data.dtype, np.float32))
        assert_(data.dtype.byteorder == '>' or (sys.byteorder == 'big' and data.dtype.byteorder == '='))
        assert_equal(data.shape, (441, 2))
        del data