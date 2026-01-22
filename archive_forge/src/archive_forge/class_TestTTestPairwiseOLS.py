from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import (
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning
class TestTTestPairwiseOLS(CheckPairwise):

    @classmethod
    def setup_class(cls):
        from statsmodels.formula.api import ols
        import statsmodels.stats.tests.test_anova as ttmod
        test = ttmod.TestAnova3()
        test.setup_class()
        cls.data = test.data.drop([0, 1, 2])
        mod = ols('np.log(Days+1) ~ C(Duration) + C(Weight)', cls.data)
        cls.res = mod.fit()
        cls.term_name = 'C(Weight)'
        cls.constraints = ['C(Weight)[T.2]', 'C(Weight)[T.3]', 'C(Weight)[T.3] - C(Weight)[T.2]']

    def test_alpha(self):
        pw1 = self.res.t_test_pairwise(self.term_name, method='hommel', factor_labels='A B C'.split())
        pw2 = self.res.t_test_pairwise(self.term_name, method='hommel', alpha=0.01)
        assert_allclose(pw1.result_frame.iloc[:, :7].values, pw2.result_frame.iloc[:, :7].values, rtol=1e-10)
        assert_equal(pw1.result_frame.iloc[:, -1].values, [True] * 3)
        assert_equal(pw2.result_frame.iloc[:, -1].values, [False, True, False])
        assert_equal(pw1.result_frame.index.values, np.array(['B-A', 'C-A', 'C-B'], dtype=object))