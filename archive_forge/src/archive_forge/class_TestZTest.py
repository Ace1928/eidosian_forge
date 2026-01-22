import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
class TestZTest:

    @classmethod
    def setup_class(cls):
        cls.x1 = np.array([7.8, 6.6, 6.5, 7.4, 7.3, 7.0, 6.4, 7.1, 6.7, 7.6, 6.8])
        cls.x2 = np.array([4.5, 5.4, 6.1, 6.1, 5.4, 5.0, 4.1, 5.5])
        cls.d1 = DescrStatsW(cls.x1)
        cls.d2 = DescrStatsW(cls.x2)
        cls.cm = CompareMeans(cls.d1, cls.d2)

    def test(self):
        x1, x2 = (self.x1, self.x2)
        cm = self.cm
        for tc in [ztest_, ztest_smaller, ztest_larger, ztest_mu, ztest_smaller_mu, ztest_larger_mu]:
            zstat, pval = ztest(x1, x2, value=tc.null_value, alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)
            zstat, pval = cm.ztest_ind(value=tc.null_value, alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)
            tc_conf_int = tc.conf_int.copy()
            if np.isnan(tc_conf_int[0]):
                tc_conf_int[0] = -np.inf
            if np.isnan(tc_conf_int[1]):
                tc_conf_int[1] = np.inf
            ci = zconfint(x1, x2, value=0, alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)
            ci = cm.zconfint_diff(alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)
            ci = zconfint(x1, x2, value=tc.null_value, alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int - tc.null_value, rtol=1e-10)
        for tc in [ztest_unequal, ztest_smaller_unequal, ztest_larger_unequal]:
            zstat, pval = ztest(x1, x2, value=tc.null_value, alternative=alternatives[tc.alternative], usevar='unequal')
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)
        d1 = self.d1
        for tc in [ztest_mu_1s, ztest_smaller_mu_1s, ztest_larger_mu_1s]:
            zstat, pval = ztest(x1, value=tc.null_value, alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)
            zstat, pval = d1.ztest_mean(value=tc.null_value, alternative=alternatives[tc.alternative])
            assert_allclose(zstat, tc.statistic, rtol=1e-10)
            assert_allclose(pval, tc.p_value, rtol=1e-10, atol=1e-16)
            tc_conf_int = tc.conf_int.copy()
            if np.isnan(tc_conf_int[0]):
                tc_conf_int[0] = -np.inf
            if np.isnan(tc_conf_int[1]):
                tc_conf_int[1] = np.inf
            ci = zconfint(x1, value=0, alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)
            ci = d1.zconfint_mean(alternative=alternatives[tc.alternative])
            assert_allclose(ci, tc_conf_int, rtol=1e-10)