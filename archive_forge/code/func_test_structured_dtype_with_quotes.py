import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_structured_dtype_with_quotes():
    data = StringIO("1000;2.4;'alpha';-34\n2000;3.1;'beta';29\n3500;9.9;'gamma';120\n4090;8.1;'delta';0\n5001;4.4;'epsilon';-99\n6543;7.8;'omega';-1\n")
    dtype = np.dtype([('f0', np.uint16), ('f1', np.float64), ('f2', 'S7'), ('f3', np.int8)])
    expected = np.array([(1000, 2.4, 'alpha', -34), (2000, 3.1, 'beta', 29), (3500, 9.9, 'gamma', 120), (4090, 8.1, 'delta', 0), (5001, 4.4, 'epsilon', -99), (6543, 7.8, 'omega', -1)], dtype=dtype)
    res = np.loadtxt(data, dtype=dtype, delimiter=';', quotechar="'")
    assert_array_equal(res, expected)