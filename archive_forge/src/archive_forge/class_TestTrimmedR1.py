import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pytest
from statsmodels.stats.robust_compare import (
import statsmodels.stats.oneway as smo
from statsmodels.tools.testing import Holder
from scipy.stats import trim1
class TestTrimmedR1:

    @classmethod
    def setup_class(cls):
        x = np.array([77, 87, 88, 114, 151, 210, 219, 246, 253, 262, 296, 299, 306, 376, 428, 515, 666, 1310, 2611])
        cls.get_results()
        cls.tm = TrimmedMean(x, cls.k / 19)

    @classmethod
    def get_results(cls):
        cls.k = 1
        cls.res_basic = np.array([342.705882352941, 92.3342348150314, 380.157894736842, 92.9416968861829, 129679.029239766])
        ytt1 = Holder()
        ytt1.statistic = 3.71157981694944
        ytt1.parameter = 16
        ytt1.p_value = 0.00189544440273015
        ytt1.conf_int = np.array([146.966048669017, 538.445716036866])
        ytt1.estimate = 342.705882352941
        ytt1.null_value = 0
        ytt1.alternative = 'two.sided'
        ytt1.method = 'One sample Yuen test, trim=0.0526315789473684'
        ytt1.data_name = 'x'
        cls.ytt1 = ytt1

    def test_basic(self):
        tm = self.tm
        assert_equal(tm.nobs, 19)
        assert_equal(tm.nobs_reduced, 17)
        assert_equal(tm.fraction, self.k / 19)
        assert_equal(tm.data_trimmed.shape[0], tm.nobs_reduced)
        res = [tm.mean_trimmed, tm.std_mean_trimmed, tm.mean_winsorized, tm.std_mean_winsorized, tm.var_winsorized]
        assert_allclose(res, self.res_basic, rtol=1e-15)

    def test_inference(self):
        ytt1 = self.ytt1
        tm = self.tm
        ttt = tm.ttest_mean()
        assert_allclose(ttt[0], ytt1.statistic, rtol=1e-13)
        assert_allclose(ttt[1], ytt1.p_value, rtol=1e-13)
        assert_equal(ttt[2], ytt1.parameter)
        assert_allclose(tm.mean_trimmed, ytt1.estimate, rtol=1e-13)
        ttw_statistic, ttw_pvalue, tt_w_df = (4.090283559190728, 0.0008537789444194812, 16)
        ttw = tm.ttest_mean(transform='winsorized')
        assert_allclose(ttw[0], ttw_statistic, rtol=1e-13)
        assert_allclose(ttw[1], ttw_pvalue, rtol=1e-13)
        assert_equal(ttw[2], tt_w_df)

    def test_other(self):
        tm = self.tm
        tm2 = tm.reset_fraction(0.0)
        assert_equal(tm2.nobs_reduced, tm2.nobs)

    @pytest.mark.parametrize('axis', [0, 1])
    def test_vectorized(self, axis):
        tm = self.tm
        x = tm.data
        x2 = np.column_stack((x, 2 * x))
        if axis == 0:
            tm2d = TrimmedMean(x2, self.k / 19, axis=0)
        else:
            tm2d = TrimmedMean(x2.T, self.k / 19, axis=1)
        t1 = [tm.mean_trimmed, 2 * tm.mean_trimmed]
        assert_allclose(tm2d.mean_trimmed, t1, rtol=1e-13)
        t1 = [tm.var_winsorized, 4 * tm.var_winsorized]
        assert_allclose(tm2d.var_winsorized, t1, rtol=1e-13)
        t1 = [tm.std_mean_trimmed, 2 * tm.std_mean_trimmed]
        assert_allclose(tm2d.std_mean_trimmed, t1, rtol=1e-13)
        t1 = [tm.mean_winsorized, 2 * tm.mean_winsorized]
        assert_allclose(tm2d.mean_winsorized, t1, rtol=1e-13)
        t1 = [tm.std_mean_winsorized, 2 * tm.std_mean_winsorized]
        assert_allclose(tm2d.std_mean_winsorized, t1, rtol=1e-13)
        s2, pv2, df2 = tm2d.ttest_mean()
        s, pv, df = tm.ttest_mean()
        assert_allclose(s2, [s, s], rtol=1e-13)
        assert_allclose(pv2, [pv, pv], rtol=1e-13)
        assert_allclose(df2, df, rtol=1e-13)
        s2, pv2, df2 = tm2d.ttest_mean(transform='winsorized')
        s, pv, df = tm.ttest_mean(transform='winsorized')
        assert_allclose(s2, [s, s], rtol=1e-13)
        assert_allclose(pv2, [pv, pv], rtol=1e-13)
        assert_allclose(df2, df, rtol=1e-13)