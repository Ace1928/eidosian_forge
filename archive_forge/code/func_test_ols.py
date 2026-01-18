import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
def test_ols(self):
    res_ols = ols('dens ~ temp + I(temp ** 2.0)', self.df).fit(use_t=False)
    y = res_ols.model.endog
    xx = res_ols.model.exog
    mq_norm = norms.MQuantileNorm(0.5, norms.LeastSquares())
    mod_rlm = RLM(y, xx, M=mq_norm)
    res_rlm = mod_rlm.fit()
    assert_allclose(res_rlm.params, res_ols.params, rtol=1e-10)
    assert_allclose(res_rlm.bse, res_ols.bse, rtol=1e-10)
    assert_allclose(res_rlm.pvalues, res_ols.pvalues, rtol=1e-10)