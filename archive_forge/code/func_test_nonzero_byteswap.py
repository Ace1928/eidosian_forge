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
def test_nonzero_byteswap(self):
    a = np.array([2147483648, 128, 0], dtype=np.uint32)
    a.dtype = np.float32
    assert_equal(a.nonzero()[0], [1])
    a = a.byteswap().newbyteorder()
    assert_equal(a.nonzero()[0], [1])