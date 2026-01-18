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
def test_nonbool_logical(self):
    size = 100
    a = np.frombuffer(b'\x01' * size, dtype=np.bool_)
    b = np.frombuffer(b'\x80' * size, dtype=np.bool_)
    expected = np.ones(size, dtype=np.bool_)
    assert_array_equal(np.logical_and(a, b), expected)