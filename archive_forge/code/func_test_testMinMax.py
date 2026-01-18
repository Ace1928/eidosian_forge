from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_testMinMax(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf, s = self.d
    xr = np.ravel(x)
    xmr = ravel(xm)
    assert_(eq(max(xr), maximum.reduce(xmr)))
    assert_(eq(min(xr), minimum.reduce(xmr)))