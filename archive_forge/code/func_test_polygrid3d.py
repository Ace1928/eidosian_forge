from functools import reduce
import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
def test_polygrid3d(self):
    x1, x2, x3 = self.x
    y1, y2, y3 = self.y
    tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
    res = poly.polygrid3d(x1, x2, x3, self.c3d)
    assert_almost_equal(res, tgt)
    z = np.ones((2, 3))
    res = poly.polygrid3d(z, z, z, self.c3d)
    assert_(res.shape == (2, 3) * 3)