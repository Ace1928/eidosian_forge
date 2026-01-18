import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_read_early_eof():
    for mmap in [False, True]:
        filename = 'test-44100Hz-le-1ch-4bytes-early-eof-no-data.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='Unexpected end of file.'):
                wavfile.read(fp, mmap=mmap)