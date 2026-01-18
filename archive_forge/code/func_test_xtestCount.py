from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_xtestCount(self):
    ott = array([0.0, 1.0, 2.0, 3.0], mask=[1, 0, 0, 0])
    assert_(count(ott).dtype.type is np.intp)
    assert_equal(3, count(ott))
    assert_equal(1, count(1))
    assert_(eq(0, array(1, mask=[1])))
    ott = ott.reshape((2, 2))
    assert_(count(ott).dtype.type is np.intp)
    assert_(isinstance(count(ott, 0), np.ndarray))
    assert_(count(ott).dtype.type is np.intp)
    assert_(eq(3, count(ott)))
    assert_(getmask(count(ott, 0)) is nomask)
    assert_(eq([1, 2], count(ott, 0)))