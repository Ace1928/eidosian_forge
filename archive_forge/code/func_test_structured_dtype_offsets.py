import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_structured_dtype_offsets():
    dt = np.dtype('i1, i4, i1, i4, i1, i4', align=True)
    data = StringIO('1,2,3,4,5,6\n7,8,9,10,11,12\n')
    expected = np.array([(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)], dtype=dt)
    assert_array_equal(np.loadtxt(data, delimiter=',', dtype=dt), expected)