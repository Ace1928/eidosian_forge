import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
def test_twosample_poisson_r():
    from .results.results_rates import res_pexact_cond, res_pexact_cond_midp
    count1, n1, count2, n2 = (60, 51477.5, 30, 54308.7)
    res2 = res_pexact_cond
    res1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond')
    assert_allclose(res1.pvalue, res2.p_value, rtol=1e-13)
    assert_allclose(res1.ratio, res2.estimate, rtol=1e-13)
    assert_equal(res1.ratio_null, res2.null_value)
    res2 = res_pexact_cond_midp
    res1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp')
    assert_allclose(res1.pvalue, res2.p_value, rtol=0, atol=5e-06)
    assert_allclose(res1.ratio, res2.estimate, rtol=1e-13)
    assert_equal(res1.ratio_null, res2.null_value)
    pv2 = 0.9949053964701466
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp', value=1.2, alternative='smaller')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)
    pv2 = 0.005094603529853279
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp', value=1.2, alternative='larger')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)
    pv2 = 0.006651774552714537
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond', value=1.2, alternative='larger')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)
    pv2 = 0.9964625674930079
    rest = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond', value=1.2, alternative='smaller')
    assert_allclose(rest.pvalue, pv2, rtol=1e-12)