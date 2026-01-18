import os
import numpy as np
from numpy.testing import (
def test_refcount_vectorize(self):

    def p(x, y):
        return 123
    v = np.vectorize(p)
    _assert_valid_refcount(v)