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
def test_ufunc_casting_out(self):
    a = np.array(1.0, dtype=np.float32)
    b = np.array(1.0, dtype=np.float64)
    c = np.array(1.0, dtype=np.float32)
    np.add(a, b, out=c)
    assert_equal(c, 2.0)