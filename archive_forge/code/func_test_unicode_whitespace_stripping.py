import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'] + 'efdFD' + '?')
def test_unicode_whitespace_stripping(dtype):
    txt = StringIO(' 3 ,"\u202f2\n"')
    res = np.loadtxt(txt, dtype=dtype, delimiter=',', quotechar='"')
    assert_array_equal(res, np.array([3, 2]).astype(dtype))