import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_step_direction(self):

    def f(x):
        y = np.exp(x)
        y[(x < 0) + (x > 2)] = np.nan
        return y
    x = np.linspace(0, 2, 10)
    step_direction = np.zeros_like(x)
    step_direction[x < 0.6], step_direction[x > 1.4] = (1, -1)
    res = zeros._differentiate(f, x, step_direction=step_direction)
    assert_allclose(res.df, np.exp(x))
    assert np.all(res.success)