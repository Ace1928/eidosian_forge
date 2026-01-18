import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO
import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY
@pytest.mark.parametrize('skiprows', (2, 3))
def test_warn_on_skipped_data(skiprows):
    data = '1 2 3\n4 5 6'
    txt = StringIO(data)
    with pytest.warns(UserWarning, match='input contained no data'):
        np.loadtxt(txt, skiprows=skiprows)