import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats.contingency import relative_risk
@pytest.mark.parametrize('ec, et, cc, ct', [(0, 0, 10, 20), (-1, 10, 1, 5), (1, 10, 0, 0), (1, 10, -1, 4)])
def test_relative_risk_bad_value(ec, et, cc, ct):
    with pytest.raises(ValueError, match='must be an integer not less than'):
        relative_risk(ec, et, cc, ct)