import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import powm1
@pytest.mark.parametrize('x, y, expected', [(0.0, 0.0, 0.0), (0.0, -1.5, np.inf), (0.0, 1.75, -1.0), (-1.5, 2.0, 1.25), (-1.5, 3.0, -4.375), (np.nan, 0.0, 0.0), (1.0, np.nan, 0.0), (1.0, np.inf, 0.0), (1.0, -np.inf, 0.0), (np.inf, 7.5, np.inf), (np.inf, -7.5, -1.0), (3.25, np.inf, np.inf), (np.inf, np.inf, np.inf), (np.inf, -np.inf, -1.0), (np.inf, 0.0, 0.0), (-np.inf, 0.0, 0.0), (-np.inf, 2.0, np.inf), (-np.inf, 3.0, -np.inf), (-1.0, float(2 ** 53 - 1), -2.0)])
def test_powm1_exact_cases(x, y, expected):
    p = powm1(x, y)
    assert p == expected