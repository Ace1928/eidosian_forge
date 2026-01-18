import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_periodic_eval(self):
    x = np.linspace(0, 2 * np.pi, 10)
    y = np.cos(x)
    S = CubicSpline(x, y, bc_type='periodic')
    assert_almost_equal(S(1), S(1 + 2 * np.pi), decimal=15)