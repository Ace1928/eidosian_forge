import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from scipy.special import lambertw
from numpy import nan, inf, pi, e, isnan, log, r_, array, complex128
from scipy.special._testutils import FuncData
def test_ufunc():
    assert_array_almost_equal(lambertw(r_[0.0, e, 1.0]), r_[0.0, 1.0, 0.5671432904097838])