import os
import numpy as np
from numpy.testing import (
def test_poly_div(self):
    u = np.poly1d([1, 2, 3])
    v = np.poly1d([1, 2, 3, 4, 5])
    q, r = np.polydiv(u, v)
    assert_equal(q * v + r, u)