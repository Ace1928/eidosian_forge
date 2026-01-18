import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats.contingency import relative_risk
def test_relative_risk_confidence_interval():
    result = relative_risk(exposed_cases=16, exposed_total=128, control_cases=24, control_total=256)
    rr = result.relative_risk
    ci = result.confidence_interval(confidence_level=0.95)
    assert_allclose(rr, 4 / 3)
    assert_allclose((ci.low, ci.high), (0.7347317, 2.419628), rtol=5e-07)