import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('q', ('"', "'", '`'))
def test_quoted_field(q):
    txt = StringIO(f'{q}alpha, x{q}, 2.5\n{q}beta, y{q}, 4.5\n{q}gamma, z{q}, 5.0\n')
    dtype = np.dtype([('f0', 'U8'), ('f1', np.float64)])
    expected = np.array([('alpha, x', 2.5), ('beta, y', 4.5), ('gamma, z', 5.0)], dtype=dtype)
    res = np.loadtxt(txt, dtype=dtype, delimiter=',', quotechar=q)
    assert_array_equal(res, expected)