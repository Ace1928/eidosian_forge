import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
from scipy.stats import variation
from scipy._lib._util import AxisError
@pytest.mark.parametrize('nan_policy, expected', [('propagate', np.nan), ('omit', np.sqrt(20 / 3) / 4)])
def test_variation_nan(self, nan_policy, expected):
    x = np.arange(10.0)
    x[9] = np.nan
    assert_allclose(variation(x, nan_policy=nan_policy), expected)