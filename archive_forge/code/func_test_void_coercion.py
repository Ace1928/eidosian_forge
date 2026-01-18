import os
import numpy as np
from numpy.testing import (
def test_void_coercion(self):
    dt = np.dtype([('a', 'f4'), ('b', 'i4')])
    x = np.zeros((1,), dt)
    assert_(np.r_[x, x].dtype == dt)