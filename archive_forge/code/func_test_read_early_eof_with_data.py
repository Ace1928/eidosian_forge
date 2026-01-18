import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_read_early_eof_with_data():
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes-early-eof.wav'
        with open(datafile(filename), 'rb') as fp:
            with warns(wavfile.WavFileWarning, match='Reached EOF'):
                rate, data = wavfile.read(fp, mmap=mmap)
                assert data.size > 0
                assert rate == 44100
                data[0] = 0