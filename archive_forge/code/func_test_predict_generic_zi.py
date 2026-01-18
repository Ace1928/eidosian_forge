from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
def test_predict_generic_zi(self):
    res = self.res
    endog = self.endog
    exog = self.res.model.exog
    prob_infl = self.prob_infl
    nobs = len(endog)
    freq = np.bincount(endog.astype(int)) / len(endog)
    probs = res.predict(which='prob')
    probsm = probs.mean(0)
    assert_allclose(freq, probsm, atol=0.02)
    probs_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='prob')
    probs_unique2 = probs[[1, nobs - 1]]
    assert_allclose(probs_unique, probs_unique2, atol=1e-10)
    probs0_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='prob-zero')
    assert_allclose(probs0_unique, probs_unique2[:, 0], rtol=1e-10)
    probs_main_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='prob-main')
    probs_main = res.predict(which='prob-main')
    probs_main[[0, -1]]
    assert_allclose(probs_main_unique, probs_main[[0, -1]], rtol=1e-10)
    assert_allclose(probs_main_unique, 1 - prob_infl, atol=0.01)
    pred = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]))
    pred1 = (endog[exog[:, 1] == 0].mean(), endog[exog[:, 1] == 1].mean())
    assert_allclose(pred, pred1, rtol=0.05)
    pred_main_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='mean-main')
    assert_allclose(pred_main_unique, np.exp(np.cumsum(res.params[1:3])), rtol=1e-10)
    mean_nz = (endog[(exog[:, 1] == 0) & (endog > 0)].mean(), endog[(exog[:, 1] == 1) & (endog > 0)].mean())
    pred_nonzero_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='mean-nonzero')
    assert_allclose(pred_nonzero_unique, mean_nz, rtol=0.05)
    pred_lin_unique = res.predict(exog=[[1, 0], [1, 1]], exog_infl=np.asarray([[1], [1]]), which='linear')
    assert_allclose(pred_lin_unique, np.cumsum(res.params[1:3]), rtol=1e-10)