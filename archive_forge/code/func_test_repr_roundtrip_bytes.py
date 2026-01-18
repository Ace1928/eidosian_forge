import warnings
import platform
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.tests._locales import CommaDecimalPointLocale
@pytest.mark.skipif(string_to_longdouble_inaccurate, reason='Need strtold_l')
def test_repr_roundtrip_bytes():
    o = 1 + LD_INFO.eps
    assert_equal(np.longdouble(repr(o).encode('ascii')), o)