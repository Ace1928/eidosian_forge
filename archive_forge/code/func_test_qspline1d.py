import numpy as np
from numpy import array
from numpy.testing import (assert_allclose, assert_array_equal,
import pytest
from pytest import raises
import scipy.signal._bsplines as bsp
from scipy import signal
def test_qspline1d(self):
    np.random.seed(12463)
    assert_array_equal(bsp.qspline1d(array([0])), [0.0])
    raises(ValueError, bsp.qspline1d, array([1.0, 2, 3, 4, 5]), 1.0)
    raises(ValueError, bsp.qspline1d, array([1.0, 2, 3, 4, 5]), -1.0)
    q1d0 = array([0.85350007, 2.02441743, 2.99999534, 3.97561055, 5.14634135])
    assert_allclose(bsp.qspline1d(array([1.0, 2, 3, 4, 5])), q1d0)