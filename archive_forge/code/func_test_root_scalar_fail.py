import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_root_scalar_fail(self):
    message = 'fprime2 must be specified for halley'
    with pytest.raises(ValueError, match=message):
        root_scalar(f1, method='halley', fprime=f1_1, x0=3, xtol=1e-06)
    message = 'fprime must be specified for halley'
    with pytest.raises(ValueError, match=message):
        root_scalar(f1, method='halley', fprime2=f1_2, x0=3, xtol=1e-06)