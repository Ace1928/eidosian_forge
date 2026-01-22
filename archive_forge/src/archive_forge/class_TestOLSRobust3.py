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
class TestOLSRobust3(TestOLSRobust1):

    def setup_method(self):
        res_ols = self.res1
        self.bse_robust = res_ols.HC0_se
        self.cov_robust = res_ols.cov_HC0
        self.small = False
        self.res2 = res.results_ivhc0_large