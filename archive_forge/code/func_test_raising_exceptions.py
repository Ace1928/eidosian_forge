import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def test_raising_exceptions(self):

    def myfunc(x):
        raise RuntimeError('myfunc')

    def myfunc1(x):
        return optimize.rosen(x)

    def callback(x):
        raise ValueError('callback')
    with pytest.raises(RuntimeError):
        optimize.minimize(myfunc, [0, 1], method='TNC')
    with pytest.raises(ValueError):
        optimize.minimize(myfunc1, [0, 1], method='TNC', callback=callback)