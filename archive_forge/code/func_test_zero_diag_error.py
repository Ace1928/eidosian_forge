import numpy as np
from scipy.linalg._solve_toeplitz import levinson
from scipy.linalg import solve, toeplitz, solve_toeplitz
from numpy.testing import assert_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
def test_zero_diag_error():
    random = np.random.RandomState(1234)
    n = 4
    c = random.randn(n)
    r = random.randn(n)
    y = random.randn(n)
    c[0] = 0
    assert_raises(np.linalg.LinAlgError, solve_toeplitz, (c, r), b=y)