import sys
import sysconfig
import weakref
from pathlib import Path
import pytest
import numpy as np
from numpy.ctypeslib import ndpointer, load_library, as_array
from numpy.testing import assert_, assert_array_equal, assert_raises, assert_equal
def test_subarray(self):
    dt = np.dtype((np.int32, (2, 3)))
    ct = np.ctypeslib.as_ctypes_type(dt)
    assert_equal(ct, 2 * (3 * ctypes.c_int32))