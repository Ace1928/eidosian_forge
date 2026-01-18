from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polyval3d(self):
    x1, x2, x3 = self.x
    y1, y2, y3 = self.y
    assert_raises_regex(ValueError, 'incompatible', poly.polyval3d, x1, x2, x3[:2], self.c3d)
    tgt = y1 * y2 * y3
    res = poly.polyval3d(x1, x2, x3, self.c3d)
    assert_almost_equal(res, tgt)
    z = np.ones((2, 3))
    res = poly.polyval3d(z, z, z, self.c3d)
    assert_(res.shape == (2, 3))