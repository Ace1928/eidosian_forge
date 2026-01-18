import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.stats.proportion as smprop
from statsmodels.stats.proportion import (
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.stats.tests.results.results_proportion import res_binom, res_binom_methods
def test_test_2indep():
    alpha = 0.05
    count1, nobs1 = (7, 34)
    count2, nobs2 = (1, 34)
    methods_both = [('diff', 'agresti-caffo'), ('diff', 'score'), ('diff', 'wald'), ('ratio', 'log'), ('ratio', 'log-adjusted'), ('ratio', 'score'), ('odds-ratio', 'logit'), ('odds-ratio', 'logit-adjusted'), ('odds-ratio', 'logit-smoothed'), ('odds-ratio', 'score')]
    for co, method in methods_both:
        low, upp = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare=co, method=method, alpha=alpha, correction=False)
        res = smprop.test_proportions_2indep(count1, nobs1, count2, nobs2, value=low, compare=co, method=method, correction=False)
        assert_allclose(res.pvalue, alpha, atol=1e-10)
        res = smprop.test_proportions_2indep(count1, nobs1, count2, nobs2, value=upp, compare=co, method=method, correction=False)
        assert_allclose(res.pvalue, alpha, atol=1e-10)
        _, pv = smprop.test_proportions_2indep(count1, nobs1, count2, nobs2, value=upp, compare=co, method=method, alternative='smaller', correction=False, return_results=False)
        assert_allclose(pv, alpha / 2, atol=1e-10)
        _, pv = smprop.test_proportions_2indep(count1, nobs1, count2, nobs2, value=low, compare=co, method=method, alternative='larger', correction=False, return_results=False)
        assert_allclose(pv, alpha / 2, atol=1e-10)
    co, method = ('ratio', 'score')
    low, upp = confint_proportions_2indep(count1, nobs1, count2, nobs2, compare=co, method=method, alpha=alpha, correction=True)
    res = smprop.test_proportions_2indep(count1, nobs1, count2, nobs2, value=low, compare=co, method=method, correction=True)
    assert_allclose(res.pvalue, alpha, atol=1e-10)