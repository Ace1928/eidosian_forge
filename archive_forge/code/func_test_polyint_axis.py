from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyint_axis(self):
    c2d = np.random.random((3, 4))
    tgt = np.vstack([poly.polyint(c) for c in c2d.T]).T
    res = poly.polyint(c2d, axis=0)
    assert_almost_equal(res, tgt)
    tgt = np.vstack([poly.polyint(c) for c in c2d])
    res = poly.polyint(c2d, axis=1)
    assert_almost_equal(res, tgt)
    tgt = np.vstack([poly.polyint(c, k=3) for c in c2d])
    res = poly.polyint(c2d, k=3, axis=1)
    assert_almost_equal(res, tgt)