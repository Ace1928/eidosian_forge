import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
def test_unsigned_max(self):
    types = np.sctypes['uint']
    for T in types:
        with np.errstate(over='ignore'):
            max_calculated = T(0) - T(1)
        assert_equal(iinfo(T).max, max_calculated)