import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_null_character():
    res = np.loadtxt(['1\x002\x003\n', '4\x005\x006'], delimiter='\x00')
    assert_array_equal(res, [[1, 2, 3], [4, 5, 6]])
    res = np.loadtxt(['1\x00,2\x00,3\n', '4\x00,5\x00,6'], delimiter=',', dtype=object)
    assert res.tolist() == [['1\x00', '2\x00', '3'], ['4\x00', '5\x00', '6']]