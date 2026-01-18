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
def test_searchsorted_wrong_dtype(self):
    a = np.array([('a', 1)], dtype='S1, int')
    assert_raises(TypeError, np.searchsorted, a, 1.2)
    dtype = np.format_parser(['i4', 'i4'], [], [])
    a = np.recarray((2,), dtype)
    a[...] = [(1, 2), (3, 4)]
    assert_raises(TypeError, np.searchsorted, a, 1)