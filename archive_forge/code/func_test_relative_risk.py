import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats.contingency import relative_risk
@pytest.mark.parametrize('exposed_cases, exposed_total, control_cases, control_total, expected_rr', [(1, 4, 3, 8, 0.25 / 0.375), (0, 10, 5, 20, 0), (0, 10, 0, 20, np.nan), (5, 15, 0, 20, np.inf)])
def test_relative_risk(exposed_cases, exposed_total, control_cases, control_total, expected_rr):
    result = relative_risk(exposed_cases, exposed_total, control_cases, control_total)
    assert_allclose(result.relative_risk, expected_rr, rtol=1e-13)