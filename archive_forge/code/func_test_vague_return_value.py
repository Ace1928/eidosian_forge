import sys
import sysconfig
import weakref
from pathlib import Path
import pytest
import numpy as np
from numpy.ctypeslib import ndpointer, load_library, as_array
from numpy.testing import assert_, assert_array_equal, assert_raises, assert_equal
def test_vague_return_value(self):
    """ Test that vague ndpointer return values do not promote to arrays """
    arr = np.zeros((2, 3))
    ptr_type = ndpointer(dtype=arr.dtype)
    c_forward_pointer.restype = ptr_type
    c_forward_pointer.argtypes = (ptr_type,)
    ret = c_forward_pointer(arr)
    assert_(isinstance(ret, ptr_type))