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
@pytest.mark.skipif(IS_PYSTON, reason='Pyston disables recursion checking')
def test_object_array_circular_reference(self):
    a = np.array(0, dtype=object)
    b = np.array(0, dtype=object)
    a[()] = b
    b[()] = a
    assert_raises(RecursionError, int, a)
    a[()] = None
    a = np.array(0, dtype=object)
    a[...] += 1
    assert_equal(a, 1)