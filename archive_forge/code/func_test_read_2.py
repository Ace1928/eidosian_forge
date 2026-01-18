import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_read_2():
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-2ch-1byteu.wav'
        rate, data = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 8000)
        assert_(np.issubdtype(data.dtype, np.uint8))
        assert_equal(data.shape, (800, 2))
        del data