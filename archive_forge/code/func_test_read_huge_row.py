import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_read_huge_row():
    row = '1.5, 2.5,' * 50000
    row = row[:-1] + '\n'
    txt = StringIO(row * 2)
    res = np.loadtxt(txt, delimiter=',', dtype=float)
    assert_equal(res, np.tile([1.5, 2.5], (2, 50000)))