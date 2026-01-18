import sys
import os
import warnings
import pytest
from io import BytesIO
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.lib import format
def test_version_2_0():
    f = BytesIO()
    dt = [('%d' % i * 100, float) for i in range(500)]
    d = np.ones(1000, dtype=dt)
    format.write_array(f, d, version=(2, 0))
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', UserWarning)
        format.write_array(f, d)
        assert_(w[0].category is UserWarning)
    f.seek(0)
    header = f.readline()
    assert_(len(header) % format.ARRAY_ALIGN == 0)
    f.seek(0)
    n = format.read_array(f, max_header_size=200000)
    assert_array_equal(d, n)
    assert_raises(ValueError, format.write_array, f, d, (1, 0))