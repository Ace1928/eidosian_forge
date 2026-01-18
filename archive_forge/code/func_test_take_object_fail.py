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
def test_take_object_fail(self):
    d = 123.0
    a = np.array([d, 1], dtype=object)
    if HAS_REFCOUNT:
        ref_d = sys.getrefcount(d)
    try:
        a.take([0, 100])
    except IndexError:
        pass
    if HAS_REFCOUNT:
        assert_(ref_d == sys.getrefcount(d))