import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_read_unknown_wave_format():
    for mmap in [False, True]:
        filename = 'test-8000Hz-le-1ch-1byte-ulaw.wav'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match='Unknown wave file format.*MULAW.*Supported formats'):
                wavfile.read(fp, mmap=mmap)