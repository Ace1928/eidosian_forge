import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
import scipy.special as sc
from scipy.special._testutils import assert_func_equal
@pytest.mark.parametrize('x, desired', [(-np.inf, 0), (np.inf, np.inf)])
def test_wrightomega_real_infinities(x, desired):
    assert sc.wrightomega(x) == desired