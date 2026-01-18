import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
def test_twosample_poisson():
    count1, n1, count2, n2 = (60, 51477.5, 30, 54308.7)
    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald')
    pv1r = 0.000356
    assert_allclose(pv1, pv1r * 2, rtol=0, atol=5e-06)
    assert_allclose(s1, 3.384913, atol=0, rtol=5e-06)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score')
    pv2r = 0.000316
    assert_allclose(pv2, pv2r * 2, rtol=0, atol=5e-06)
    assert_allclose(s2, 3.417402, atol=0, rtol=5e-06)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald-log')
    pv2r = 0.00042
    assert_allclose(pv2, pv2r * 2, rtol=0, atol=5e-06)
    assert_allclose(s2, 3.3393, atol=0, rtol=5e-06)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score-log')
    pv2r = 0.0002
    assert_allclose(pv2, pv2r * 2, rtol=0, atol=5e-06)
    assert_allclose(s2, 3.5406, atol=0, rtol=5e-05)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt')
    pv2r = 0.000285
    assert_allclose(pv2, pv2r * 2, rtol=0, atol=5e-06)
    assert_allclose(s2, 3.445485, atol=0, rtol=5e-06)
    count1, n1, count2, n2 = (41, 28010, 15, 19017)
    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald', value=1.5)
    pv1r = 0.2309
    assert_allclose(pv1, pv1r * 2, rtol=0, atol=0.0005)
    assert_allclose(s1, 0.735447, atol=0, rtol=5e-06)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score', value=1.5)
    pv2r = 0.2398
    assert_allclose(pv2, pv2r * 2, rtol=0, atol=0.0005)
    assert_allclose(s2, 0.706631, atol=0, rtol=5e-06)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald-log', value=1.5)
    pv2r = 0.2402
    assert_allclose(pv2, pv2r * 2, rtol=0, atol=0.0005)
    assert_allclose(s2, 0.7056, atol=0, rtol=0.0005)
    with pytest.warns(FutureWarning):
        s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score-log', ratio_null=1.5)
    pv2r = 0.2303
    assert_allclose(pv2, pv2r * 2, rtol=0, atol=0.0005)
    assert_allclose(s2, 0.738, atol=0, rtol=0.0005)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt', value=1.5)
    pv2r = 0.2499
    assert_allclose(pv2, pv2r * 2, rtol=0, atol=0.005)
    assert_allclose(s2, 0.674401, atol=0, rtol=5e-06)
    count1, n1, count2, n2 = (60, 51477.5, 30, 54308.7)
    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald', alternative='larger')
    pv1r = 0.000356
    assert_allclose(pv1, pv1r, rtol=0, atol=5e-06)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score', alternative='larger')
    pv2r = 0.000316
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-06)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt', alternative='larger')
    pv2r = 0.000285
    assert_allclose(pv2, pv2r, rtol=0, atol=5e-06)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond', value=1, alternative='larger')
    pv2r = 0.000428
    assert_allclose(pv2, pv2r, rtol=0, atol=0.0005)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp', value=1, alternative='larger')
    pv2r = 0.00031
    assert_allclose(pv2, pv2r, rtol=0, atol=0.0005)
    _, pve1 = etest_poisson_2indep(count1, n1, count2, n2, method='score', alternative='larger')
    pve1r = 0.000298
    assert_allclose(pve1, pve1r, rtol=0, atol=0.0005)
    _, pve1 = etest_poisson_2indep(count1, n1, count2, n2, method='wald', alternative='larger')
    pve1r = 0.000298
    assert_allclose(pve1, pve1r, rtol=0, atol=0.0005)
    count1, n1, count2, n2 = (41, 28010, 15, 19017)
    s1, pv1 = smr.test_poisson_2indep(count1, n1, count2, n2, method='wald', value=1.5, alternative='larger')
    pv1r = 0.2309
    assert_allclose(pv1, pv1r, rtol=0, atol=0.0005)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='score', value=1.5, alternative='larger')
    pv2r = 0.2398
    assert_allclose(pv2, pv2r, rtol=0, atol=0.0005)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='sqrt', value=1.5, alternative='larger')
    pv2r = 0.2499
    assert_allclose(pv2, pv2r, rtol=0, atol=0.0005)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='exact-cond', value=1.5, alternative='larger')
    pv2r = 0.2913
    assert_allclose(pv2, pv2r, rtol=0, atol=0.0005)
    s2, pv2 = smr.test_poisson_2indep(count1, n1, count2, n2, method='cond-midp', value=1.5, alternative='larger')
    pv2r = 0.245
    assert_allclose(pv2, pv2r, rtol=0, atol=0.0005)
    _, pve2 = etest_poisson_2indep(count1, n1, count2, n2, method='score', value=1.5, alternative='larger')
    pve2r = 0.2453
    assert_allclose(pve2, pve2r, rtol=0, atol=0.0005)
    _, pve2 = etest_poisson_2indep(count1, n1, count2, n2, method='wald', value=1.5, alternative='larger')
    pve2r = 0.2453
    assert_allclose(pve2, pve2r, rtol=0, atol=0.0005)