import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
@pytest.mark.parametrize('x', [np.zeros(5), [], [1, 2, np.inf, 9]])
def test_return_nan(self, x):
    y = variation(x)
    assert_equal(y, np.nan)