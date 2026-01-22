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
class TestProportion(CheckProportionMixin):

    def setup_method(self):
        self.n_success = np.array([73, 90, 114, 75])
        self.nobs = np.array([86, 93, 136, 82])
        self.res_ppt_pvals_raw = np.array([0.00533824886503131, 0.8327574849753566, 0.1880573726722516, 0.002026764254350234, 0.1309487516334318, 0.1076118730631731])
        self.res_ppt_pvals_holm = np.array([0.02669124432515654, 0.8327574849753566, 0.4304474922526926, 0.0121605855261014, 0.4304474922526926, 0.4304474922526926])
        res_prop_test = Holder()
        res_prop_test.statistic = 11.11938768628861
        res_prop_test.parameter = 3
        res_prop_test.p_value = 0.011097511366581344
        res_prop_test.estimate = np.array([0.848837209302326, 0.967741935483871, 0.838235294117647, 0.9146341463414634]).reshape(4, 1, order='F')
        res_prop_test.null_value = 'NULL'
        res_prop_test.conf_int = 'NULL'
        res_prop_test.alternative = 'two.sided'
        res_prop_test.method = '4-sample test for equality of proportions ' + 'without continuity correction'
        res_prop_test.data_name = 'smokers2 out of patients'
        self.res_prop_test = res_prop_test
        res_prop_test_val = Holder()
        res_prop_test_val.statistic = np.array([13.20305530710751]).reshape(1, 1, order='F')
        res_prop_test_val.parameter = np.array([4]).reshape(1, 1, order='F')
        res_prop_test_val.p_value = 0.010325090041836
        res_prop_test_val.estimate = np.array([0.848837209302326, 0.967741935483871, 0.838235294117647, 0.9146341463414634]).reshape(4, 1, order='F')
        res_prop_test_val.null_value = np.array([0.9, 0.9, 0.9, 0.9]).reshape(4, 1, order='F')
        res_prop_test_val.conf_int = 'NULL'
        res_prop_test_val.alternative = 'two.sided'
        res_prop_test_val.method = '4-sample test for given proportions without continuity correction'
        res_prop_test_val.data_name = 'smokers2 out of patients, null probabilities rep(c(0.9), 4)'
        self.res_prop_test_val = res_prop_test_val
        res_prop_test_1 = Holder()
        res_prop_test_1.statistic = 2.501291989664086
        res_prop_test_1.parameter = 1
        res_prop_test_1.p_value = 0.113752943640092
        res_prop_test_1.estimate = 0.848837209302326
        res_prop_test_1.null_value = 0.9
        res_prop_test_1.conf_int = np.array([0.758364348004061, 0.9094787701686766])
        res_prop_test_1.alternative = 'two.sided'
        res_prop_test_1.method = '1-sample proportions test without continuity correction'
        res_prop_test_1.data_name = 'smokers2[1] out of patients[1], null probability 0.9'
        self.res_prop_test_1 = res_prop_test_1

    def test_default_values(self):
        count = np.array([5, 12])
        nobs = np.array([83, 99])
        stat, pval = smprop.proportions_ztest(count, nobs, value=None)
        assert_almost_equal(stat, -1.4078304151258787)
        assert_almost_equal(pval, 0.15918129181156992)

    def test_scalar(self):
        count = 5
        nobs = 83
        value = 0.05
        stat, pval = smprop.proportions_ztest(count, nobs, value=value)
        assert_almost_equal(stat, 0.392126026314)
        assert_almost_equal(pval, 0.694965098115)
        assert_raises(ValueError, smprop.proportions_ztest, count, nobs, value=None)