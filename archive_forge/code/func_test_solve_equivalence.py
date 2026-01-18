import numpy as np
from scipy.linalg._solve_toeplitz import levinson
from scipy.linalg import solve, toeplitz, solve_toeplitz
from numpy.testing import assert_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
def test_solve_equivalence():
    random = np.random.RandomState(1234)
    for n in (1, 2, 3, 10):
        c = random.randn(n)
        if random.rand() < 0.5:
            c = c + 1j * random.randn(n)
        r = random.randn(n)
        if random.rand() < 0.5:
            r = r + 1j * random.randn(n)
        y = random.randn(n)
        if random.rand() < 0.5:
            y = y + 1j * random.randn(n)
        actual = solve_toeplitz((c, r), y)
        desired = solve(toeplitz(c, r=r), y)
        assert_allclose(actual, desired)
        actual = solve_toeplitz(c, b=y)
        desired = solve(toeplitz(c), y)
        assert_allclose(actual, desired)