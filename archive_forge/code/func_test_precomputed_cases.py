import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose, assert_equal)
from scipy.linalg import polar, eigh
def test_precomputed_cases():
    for a, side, expected_u, expected_p in precomputed_cases:
        check_precomputed_polar(a, side, expected_u, expected_p)