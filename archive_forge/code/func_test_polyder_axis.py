from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyder_axis(self):
    c2d = np.random.random((3, 4))
    tgt = np.vstack([poly.polyder(c) for c in c2d.T]).T
    res = poly.polyder(c2d, axis=0)
    assert_almost_equal(res, tgt)
    tgt = np.vstack([poly.polyder(c) for c in c2d])
    res = poly.polyder(c2d, axis=1)
    assert_almost_equal(res, tgt)