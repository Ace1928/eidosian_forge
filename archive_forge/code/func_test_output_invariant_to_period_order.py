from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import MSTL
@pytest.mark.parametrize('data, periods_ordered, windows_ordered, periods_not_ordered, windows_not_ordered', [(data, (12, 24, 24 * 7), (11, 15, 19), (12, 24 * 7, 24), (11, 19, 15)), (data, (12, 24, 24 * 7 * 1000000.0), (11, 15, 19), (12, 24 * 7 * 1000000.0, 24), (11, 19, 15)), (data, (12, 24, 24 * 7), None, (12, 24 * 7, 24), None)], indirect=['data'])
def test_output_invariant_to_period_order(data, periods_ordered, windows_ordered, periods_not_ordered, windows_not_ordered):
    mod1 = MSTL(endog=data, periods=periods_ordered, windows=windows_ordered)
    res1 = mod1.fit()
    mod2 = MSTL(endog=data, periods=periods_not_ordered, windows=windows_not_ordered)
    res2 = mod2.fit()
    assert_equal(res1.observed, res2.observed)
    assert_equal(res1.trend, res2.trend)
    assert_equal(res1.seasonal, res2.seasonal)
    assert_equal(res1.resid, res2.resid)