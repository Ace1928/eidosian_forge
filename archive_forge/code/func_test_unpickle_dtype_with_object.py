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
def test_unpickle_dtype_with_object(self):
    dt = np.dtype([('x', int), ('y', np.object_), ('z', 'O')])
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        with BytesIO() as f:
            pickle.dump(dt, f, protocol=proto)
            f.seek(0)
            dt_ = pickle.load(f)
        assert_equal(dt, dt_)