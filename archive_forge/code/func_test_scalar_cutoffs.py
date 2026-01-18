import code
import platform
import pytest
import sys
from tempfile import TemporaryFile
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_MUSL
def test_scalar_cutoffs(self):

    def check(v):
        assert_equal(str(np.float64(v)), str(v))
        assert_equal(str(np.float64(v)), repr(v))
        assert_equal(repr(np.float64(v)), repr(v))
        assert_equal(repr(np.float64(v)), str(v))
    check(1.1234567890123457)
    check(0.011234567890123457)
    check(1e-05)
    check(0.0001)
    check(1000000000000000.0)
    check(1e+16)