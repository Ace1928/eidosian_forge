import os
import numpy as np
from numpy.testing import (
def test_poly1d(self):
    assert_equal(np.poly1d([1]) - np.poly1d([1, 0]), np.poly1d([-1, 1]))