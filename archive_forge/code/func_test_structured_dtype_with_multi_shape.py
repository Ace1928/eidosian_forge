import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_structured_dtype_with_multi_shape():
    dtype = np.dtype([('a', 'u1', (2, 2))])
    data = StringIO('0 1 2 3\n')
    expected = np.array([(((0, 1), (2, 3)),)], dtype=dtype)
    assert_array_equal(np.loadtxt(data, dtype=dtype), expected)