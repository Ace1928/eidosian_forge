import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
def test_regression_gh23867(self):

    class NonHashableWithDtype:
        __hash__ = None
        dtype = np.dtype('float32')
    x = NonHashableWithDtype()
    assert np.finfo(x) == np.finfo(x.dtype)