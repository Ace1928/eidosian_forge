import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats.contingency import relative_risk
def test_relative_risk_bad_type():
    with pytest.raises(TypeError, match='must be an integer'):
        relative_risk(1, 10, 2.0, 40)