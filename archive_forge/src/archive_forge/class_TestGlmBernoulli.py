import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
class TestGlmBernoulli(CheckModelResultsMixin, CheckComparisonMixin):

    @classmethod
    def setup_class(cls):
        from .results.results_glm import Lbw
        cls.res2 = Lbw()
        cls.res1 = GLM(cls.res2.endog, cls.res2.exog, family=sm.families.Binomial()).fit()
        modd = discrete.Logit(cls.res2.endog, cls.res2.exog)
        cls.resd = modd.fit(start_params=cls.res1.params * 0.9, disp=False)

    def test_score_r(self):
        res1 = self.res1
        res2 = self.res2
        st, pv, df = res1.model.score_test(res1.params, exog_extra=res1.model.exog[:, 1] ** 2)
        st_res = 0.2837680293459376
        assert_allclose(st, st_res, rtol=0.0001)
        st, pv, df = res1.model.score_test(res1.params, exog_extra=res1.model.exog[:, 0] ** 2)
        st_res = 0.6713492821514992
        assert_allclose(st, st_res, rtol=0.0001)
        select = list(range(9))
        select.pop(7)
        res1b = GLM(res2.endog, res2.exog.iloc[:, select], family=sm.families.Binomial()).fit()
        tres = res1b.model.score_test(res1b.params, exog_extra=res1.model.exog[:, -2])
        tres = np.asarray(tres[:2]).ravel()
        tres_r = (2.7864148487452, 0.0950667)
        assert_allclose(tres, tres_r, rtol=0.0001)
        cmd_r = '        data = read.csv("...statsmodels\\statsmodels\\genmod\\tests\\results\\stata_lbw_glm.csv")\n\n        data["race_black"] = data["race"] == "black"\n        data["race_other"] = data["race"] == "other"\n        mod = glm(low ~ age + lwt + race_black + race_other + smoke + ptl + ht + ui, family=binomial, data=data)\n        options(digits=16)\n        anova(mod, test="Rao")\n\n        library(statmod)\n        s = glm.scoretest(mod, data["age"]**2)\n        s**2\n        s = glm.scoretest(mod, data["lwt"]**2)\n        s**2\n        '