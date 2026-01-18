import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
def test_str_dtype_unit_discovery_with_converter():
    data = ['spam-a-lot'] * 60000 + ['XXXtis_but_a_scratch']
    expected = np.array(['spam-a-lot'] * 60000 + ['tis_but_a_scratch'], dtype='U17')
    conv = lambda s: s.strip('XXX')
    txt = StringIO('\n'.join(data))
    a = np.loadtxt(txt, dtype='U', converters=conv, encoding=None)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)
    fd, fname = mkstemp()
    os.close(fd)
    with open(fname, 'w') as fh:
        fh.write('\n'.join(data))
    a = np.loadtxt(fname, dtype='U', converters=conv, encoding=None)
    os.remove(fname)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)