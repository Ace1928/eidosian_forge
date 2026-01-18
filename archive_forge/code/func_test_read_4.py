import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_read_4():
    for mmap in [False, True]:
        with suppress_warnings() as sup:
            sup.filter(wavfile.WavFileWarning, 'Chunk .non-data. not understood, skipping it')
            filename = 'test-48000Hz-2ch-64bit-float-le-wavex.wav'
            rate, data = wavfile.read(datafile(filename), mmap=mmap)
        assert_equal(rate, 48000)
        assert_(np.issubdtype(data.dtype, np.float64))
        assert_equal(data.shape, (480, 2))
        del data