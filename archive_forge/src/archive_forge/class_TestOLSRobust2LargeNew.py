import numpy as np
from numpy.testing import (
import pytest
from scipy import stats
from statsmodels.datasets import macrodata
from statsmodels.regression.linear_model import OLS, WLS
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.sm_exceptions import InvalidTestWarning
from statsmodels.tools.tools import add_constant
from .results import (
class TestOLSRobust2LargeNew(TestOLSRobust1, CheckOLSRobustNewMixin):

    def setup_method(self):
        res_ols = self.res1.get_robustcov_results('HC0')
        res_ols.use_t = False
        self.res3 = self.res1
        self.res1 = res_ols
        self.bse_robust = res_ols.bse
        self.cov_robust = res_ols.cov_params()
        self.bse_robust2 = res_ols.HC0_se
        self.cov_robust2 = res_ols.cov_HC0
        self.small = False
        self.res2 = res.results_ivhc0_large

    @pytest.mark.skip(reason='not refactored yet for `large`')
    def test_fvalue(self):
        super().test_fvalue()

    @pytest.mark.skip(reason='not refactored yet for `large`')
    def test_confint(self):
        super().test_confint()