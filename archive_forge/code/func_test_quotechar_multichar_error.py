import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
def test_quotechar_multichar_error():
    txt = StringIO('1,2\n3,4')
    msg = '.*must be a single unicode character or None'
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, delimiter=',', quotechar="''")