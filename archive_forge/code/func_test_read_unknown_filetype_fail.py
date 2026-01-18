import os
import sys
from io import BytesIO
import numpy as np
from numpy.testing import (assert_equal, assert_, assert_array_equal,
import pytest
from pytest import raises, warns
from scipy.io import wavfile
def test_read_unknown_filetype_fail():
    for mmap in [False, True]:
        filename = 'example_1.nc'
        with open(datafile(filename), 'rb') as fp:
            with raises(ValueError, match="CDF.*'RIFF' and 'RIFX' supported"):
                wavfile.read(fp, mmap=mmap)