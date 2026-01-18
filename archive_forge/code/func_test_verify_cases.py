import numpy as np
from numpy.linalg import norm
from numpy.testing import (assert_, assert_allclose, assert_equal)
from scipy.linalg import polar, eigh
def test_verify_cases():
    for a in verify_cases:
        verify_polar(a)