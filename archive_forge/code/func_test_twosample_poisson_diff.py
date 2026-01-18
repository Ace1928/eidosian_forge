import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
@pytest.mark.parametrize('case', cases_diff_ng)
def test_twosample_poisson_diff(case):
    meth, res1, res2 = case
    count1, exposure1, count2, exposure2 = (41, 28010, 15, 19017)
    value = 0
    t = smr.test_poisson_2indep(count1, exposure1, count2, exposure2, value=value, method=meth, compare='diff', alternative='larger', etest_kwds=None)
    assert_allclose((t.statistic, t.pvalue), res1, atol=0.0006)
    value = 0.0002
    t = smr.test_poisson_2indep(count1, exposure1, count2, exposure2, value=value, method=meth, compare='diff', alternative='larger', etest_kwds=None)
    assert_allclose((t.statistic, t.pvalue), res2, atol=0.0007)