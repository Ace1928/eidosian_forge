import numpy as np
from scipy.linalg._solve_toeplitz import levinson
from scipy.linalg import solve, toeplitz, solve_toeplitz
from numpy.testing import assert_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
@pytest.mark.xfail(reason='Instability of Levinson iteration')
def test_unstable():
    random = np.random.RandomState(1234)
    n = 100
    c = 0.9 ** np.arange(n) ** 2
    y = random.randn(n)
    solution1 = solve_toeplitz(c, b=y)
    solution2 = solve(toeplitz(c), y)
    assert_allclose(solution1, solution2)