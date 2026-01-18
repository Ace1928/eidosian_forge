import sys
import sysconfig
import weakref
from pathlib import Path
import pytest
import numpy as np
from numpy.ctypeslib import ndpointer, load_library, as_array
from numpy.testing import assert_, assert_array_equal, assert_raises, assert_equal
def test_segmentation_fault(self):
    arr = np.zeros((224, 224, 3))
    c_arr = np.ctypeslib.as_ctypes(arr)
    arr_ref = weakref.ref(arr)
    del arr
    assert_(arr_ref() is not None)
    c_arr[0][0][0]