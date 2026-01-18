import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_string_truncation(self):
    for val in [True, 1234, 123.4, complex(1, 234)]:
        for tostr, dtype in [(asunicode, 'U'), (asbytes, 'S')]:
            b = np.array([val, tostr('xx')], dtype=dtype)
            assert_equal(tostr(b[0]), tostr(val))
            b = np.array([tostr('xx'), val], dtype=dtype)
            assert_equal(tostr(b[1]), tostr(val))
            b = np.array([val, tostr('xxxxxxxxxx')], dtype=dtype)
            assert_equal(tostr(b[0]), tostr(val))
            b = np.array([tostr('xxxxxxxxxx'), val], dtype=dtype)
            assert_equal(tostr(b[1]), tostr(val))