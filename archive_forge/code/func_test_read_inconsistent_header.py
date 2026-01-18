import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_read_inconsistent_header():
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-3ch-5S-24bit-inconsistent.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='header is invalid'):
                wavfile.read(fp, mmap=mmap)