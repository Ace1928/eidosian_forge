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
def test_proptest(self):
    pt = smprop.proportions_chisquare(self.n_success, self.nobs, value=None)
    assert_almost_equal(pt[0], self.res_prop_test.statistic, decimal=13)
    assert_almost_equal(pt[1], self.res_prop_test.p_value, decimal=13)
    pt = smprop.proportions_chisquare(self.n_success, self.nobs, value=self.res_prop_test_val.null_value[0])
    assert_almost_equal(pt[0], self.res_prop_test_val.statistic, decimal=13)
    assert_almost_equal(pt[1], self.res_prop_test_val.p_value, decimal=13)
    pt = smprop.proportions_chisquare(self.n_success[0], self.nobs[0], value=self.res_prop_test_1.null_value)
    assert_almost_equal(pt[0], self.res_prop_test_1.statistic, decimal=13)
    assert_almost_equal(pt[1], self.res_prop_test_1.p_value, decimal=13)