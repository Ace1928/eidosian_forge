from statsmodels.compat.pandas import testing as pdt
import os.path
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.stats.outliers_influence import MLEInfluence
class TestInfluenceBinomialGLMMLE(InfluenceCompareExact):

    @classmethod
    def setup_class(cls):
        yi = np.array([0, 2, 14, 19, 30])
        ni = 40 * np.ones(len(yi))
        xi = np.arange(1, len(yi) + 1)
        exog = np.column_stack((np.ones(len(yi)), xi))
        endog = np.column_stack((yi, ni - yi))
        res = GLM(endog, exog, family=families.Binomial()).fit()
        cls.infl1 = res.get_influence()
        cls.infl0 = MLEInfluence(res)
        cls.cd_rtol = 5e-05

    def test_looo(self):
        _check_looo(self)

    def test_r(self):
        infl1 = self.infl1
        cooks_d = [0.25220202795934726, 0.26107981497746285, 1.2898561442413239, 0.08449722285516942, 0.36362110845918005]
        hat = [0.2594393406119333, 0.3696442663244837, 0.3535768402250521, 0.38920919853579106, 0.6281303543027403]
        assert_allclose(infl1.hat_matrix_diag, hat, rtol=5e-06)
        assert_allclose(infl1.cooks_distance[0], cooks_d, rtol=1e-05)