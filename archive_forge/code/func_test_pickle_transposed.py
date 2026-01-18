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
def test_pickle_transposed(self):
    a = np.transpose(np.array([[2, 9], [7, 0], [3, 8]]))
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        with BytesIO() as f:
            pickle.dump(a, f, protocol=proto)
            f.seek(0)
            b = pickle.load(f)
        assert_array_equal(a, b)