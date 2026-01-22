import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_
import pytest
import statsmodels.stats.weightstats as smws
from statsmodels.tools.testing import Holder
class TestTostp1(CheckTostMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = tost_clinic_paired_1
        x1, x2 = (clinic[:15, 2], clinic[15:, 2])
        cls.res1 = Holder()
        res = smws.ttost_paired(x1, x2, -0.6, 0.6, transform=None)
        cls.res1.pvalue = res[0]
        res_ds = smws.DescrStatsW(x1 - x2, weights=None, ddof=0)
        cls.res1.tconfint_diff = res_ds.tconfint_mean(0.1)
        cls.res1.confint_05 = res_ds.tconfint_mean(0.05)
        cls.res1.mean_diff = res_ds.mean
        cls.res1.std_mean_diff = res_ds.std_mean
        cls.res2b = ttest_clinic_paired_1

    def test_special(self):
        assert_almost_equal(self.res1.tconfint_diff, self.res2.ci_diff, decimal=13)
        assert_almost_equal(self.res1.mean_diff, self.res2.mean_diff, decimal=13)
        assert_almost_equal(self.res1.std_mean_diff, self.res2.se_diff, decimal=13)
        assert_almost_equal(self.res1.confint_05, self.res2b.conf_int, decimal=13)