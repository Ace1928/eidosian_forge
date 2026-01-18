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
def test_void_copyswap(self):
    dt = np.dtype([('one', '<i4'), ('two', '<i4')])
    x = np.array((1, 2), dtype=dt)
    x = x.byteswap()
    assert_(x['one'] > 1 and x['two'] > 2)