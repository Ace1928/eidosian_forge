import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
def test_regression_gh23108(self):
    f1 = np.finfo(np.float32(1.0))
    f2 = np.finfo(np.float64(1.0))
    assert f1 != f2