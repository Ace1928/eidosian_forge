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
def test_proportion_ztests():
    res1 = smprop.proportions_ztest(15, 20.0, value=0.5, prop_var=0.5)
    res2 = smprop.proportions_chisquare(15, 20.0, value=0.5)
    assert_almost_equal(res1[1], res2[1], decimal=13)
    res1 = smprop.proportions_ztest(np.asarray([15, 10]), np.asarray([20.0, 20]), value=0, prop_var=None)
    res2 = smprop.proportions_chisquare(np.asarray([15, 10]), np.asarray([20.0, 20]))
    assert_almost_equal(res1[1], res2[1], decimal=13)
    res1 = smprop.proportions_ztest(np.asarray([15, 10]), np.asarray([20, 50000]), value=0, prop_var=None)
    res2 = smprop.proportions_chisquare(np.asarray([15, 10]), np.asarray([20, 50000]))
    assert_almost_equal(res1[1], res2[1], decimal=13)
    assert_array_less(0, res2[-1][1])