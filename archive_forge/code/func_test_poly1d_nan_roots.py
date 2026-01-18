import os
import numpy as np
from numpy.testing import (
def test_poly1d_nan_roots(self):
    p = np.poly1d([np.nan, np.nan, 1], r=False)
    assert_raises(np.linalg.LinAlgError, getattr, p, 'r')