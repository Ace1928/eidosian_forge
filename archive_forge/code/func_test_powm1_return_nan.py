import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.special import powm1
@pytest.mark.parametrize('x, y', [(-1.25, 751.03), (-1.25, np.inf), (np.nan, np.nan), (-np.inf, -np.inf), (-np.inf, 2.5)])
def test_powm1_return_nan(x, y):
    p = powm1(x, y)
    assert np.isnan(p)