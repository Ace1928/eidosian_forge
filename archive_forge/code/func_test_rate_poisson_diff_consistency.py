import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
@pytest.mark.parametrize('method', methods_diff)
def test_rate_poisson_diff_consistency(method):
    count1, n1, count2, n2 = (30, 400 / 10, 7, 300 / 10)
    ci = confint_poisson_2indep(count1, n1, count2, n2, method=method, compare='diff')
    pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[0], method=method, compare='diff').pvalue
    pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[1], method=method, compare='diff').pvalue
    rtol = 1e-10
    if method in ['score']:
        rtol = 1e-06
    assert_allclose(pv1, 0.05, rtol=rtol)
    assert_allclose(pv2, 0.05, rtol=rtol)
    pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[0], method=method, compare='diff', alternative='larger').pvalue
    pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, value=ci[1], method=method, compare='diff', alternative='smaller').pvalue
    assert_allclose(pv1, 0.025, rtol=rtol)
    assert_allclose(pv2, 0.025, rtol=rtol)