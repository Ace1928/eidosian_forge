import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
def test_tost_poisson():
    count1, n1, count2, n2 = (60, 51477.5, 30, 54308.7)
    low, upp = (1.33973572177265, 3.388365573616252)
    res = smr.tost_poisson_2indep(count1, n1, count2, n2, low, upp, method='exact-cond')
    assert_allclose(res.pvalue, 0.025, rtol=1e-12)
    methods = ['wald', 'score', 'sqrt', 'exact-cond', 'cond-midp']
    for meth in methods:
        res = smr.tost_poisson_2indep(count1, n1, count2, n2, low, upp, method=meth)
        assert_allclose(res.pvalue, 0.025, atol=0.01)