import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
class TestFinfo:

    def test_basic(self):
        dts = list(zip(['f2', 'f4', 'f8', 'c8', 'c16'], [np.float16, np.float32, np.float64, np.complex64, np.complex128]))
        for dt1, dt2 in dts:
            assert_finfo_equal(finfo(dt1), finfo(dt2))
        assert_raises(ValueError, finfo, 'i4')

    def test_regression_gh23108(self):
        f1 = np.finfo(np.float32(1.0))
        f2 = np.finfo(np.float64(1.0))
        assert f1 != f2

    def test_regression_gh23867(self):

        class NonHashableWithDtype:
            __hash__ = None
            dtype = np.dtype('float32')
        x = NonHashableWithDtype()
        assert np.finfo(x) == np.finfo(x.dtype)