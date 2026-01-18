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
def test_score_test_2indep():
    count1, nobs1 = (7, 34)
    count2, nobs2 = (1, 34)
    for co in ['diff', 'ratio', 'or']:
        res = score_test_proportions_2indep(count1, nobs1, count2, nobs2, compare=co)
        assert_allclose(res.prop1_null, res.prop2_null, rtol=1e-10)
        val = 0 if co == 'diff' else 1.0
        s0, pv0 = score_test_proportions_2indep(count1, nobs1, count2, nobs2, compare=co, value=val, return_results=False)[:2]
        s1, pv1 = score_test_proportions_2indep(count1, nobs1, count2, nobs2, compare=co, value=val + 1e-10, return_results=False)[:2]
        assert_allclose(s0, s1, rtol=1e-08)
        assert_allclose(pv0, pv1, rtol=1e-08)
        s1, pv1 = score_test_proportions_2indep(count1, nobs1, count2, nobs2, compare=co, value=val - 1e-10, return_results=False)[:2]
        assert_allclose(s0, s1, rtol=1e-08)
        assert_allclose(pv0, pv1, rtol=1e-08)