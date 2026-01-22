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
class TestGlmPoissonOffset(CheckModelResultsMixin):

    @classmethod
    def setup_class(cls):
        from .results.results_glm import Cpunish_offset
        cls.decimal_params = DECIMAL_4
        cls.decimal_bse = DECIMAL_4
        cls.decimal_aic_R = 3
        data = cpunish.load()
        data.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        data.exog[:, 3] = np.log(data.exog[:, 3])
        data.exog = add_constant(data.exog, prepend=True)
        exposure = [100] * len(data.endog)
        cls.data = data
        cls.exposure = exposure
        cls.res1 = GLM(data.endog, data.exog, family=sm.families.Poisson(), exposure=exposure).fit()
        cls.res2 = Cpunish_offset()

    def test_missing(self):
        endog = self.data.endog.copy()
        endog[[2, 4, 6, 8]] = np.nan
        mod = GLM(endog, self.data.exog, family=sm.families.Poisson(), exposure=self.exposure, missing='drop')
        assert_equal(mod.exposure.shape[0], 13)

    def test_offset_exposure(self):
        np.random.seed(382304)
        endog = np.random.randint(0, 10, 100)
        exog = np.random.normal(size=(100, 3))
        exposure = np.random.uniform(1, 2, 100)
        offset = np.random.uniform(1, 2, 100)
        mod1 = GLM(endog, exog, family=sm.families.Poisson(), offset=offset, exposure=exposure).fit()
        offset2 = offset + np.log(exposure)
        mod2 = GLM(endog, exog, family=sm.families.Poisson(), offset=offset2).fit()
        assert_almost_equal(mod1.params, mod2.params)
        assert_allclose(mod1.null, mod2.null, rtol=1e-10)
        mod1_ = mod1.model
        kwds = mod1_._get_init_kwds()
        assert_allclose(kwds['exposure'], exposure, rtol=1e-14)
        assert_allclose(kwds['offset'], mod1_.offset, rtol=1e-14)
        mod3 = mod1_.__class__(mod1_.endog, mod1_.exog, **kwds)
        assert_allclose(mod3.exposure, mod1_.exposure, rtol=1e-14)
        assert_allclose(mod3.offset, mod1_.offset, rtol=1e-14)
        resr1 = mod1.model.fit_regularized()
        resr2 = mod2.model.fit_regularized()
        assert_allclose(resr1.params, resr2.params, rtol=1e-10)

    def test_predict(self):
        np.random.seed(382304)
        endog = np.random.randint(0, 10, 100)
        exog = np.random.normal(size=(100, 3))
        exposure = np.random.uniform(1, 2, 100)
        mod1 = GLM(endog, exog, family=sm.families.Poisson(), exposure=exposure).fit()
        exog1 = np.random.normal(size=(10, 3))
        exposure1 = np.random.uniform(1, 2, 10)
        pred1 = mod1.predict(exog=exog1, exposure=exposure1)
        pred2 = mod1.predict(exog=exog1, exposure=2 * exposure1)
        assert_almost_equal(pred2, 2 * pred1)
        pred3 = mod1.predict()
        pred4 = mod1.predict(exposure=exposure)
        pred5 = mod1.predict(exog=exog, exposure=exposure)
        assert_almost_equal(pred3, pred4)
        assert_almost_equal(pred4, pred5)
        offset = np.random.uniform(1, 2, 100)
        mod2 = GLM(endog, exog, offset=offset, family=sm.families.Poisson()).fit()
        pred1 = mod2.predict()
        pred2 = mod2.predict(which='mean', offset=offset)
        pred3 = mod2.predict(exog=exog, which='mean', offset=offset)
        assert_almost_equal(pred1, pred2)
        assert_almost_equal(pred2, pred3)
        mod3 = GLM(endog, exog, family=sm.families.Poisson()).fit()
        offset = np.random.uniform(1, 2, 10)
        with pytest.warns(FutureWarning):
            pred1 = mod3.predict(exog=exog1, offset=offset, linear=True)
        pred2 = mod3.predict(exog=exog1, offset=2 * offset, which='linear')
        assert_almost_equal(pred2, pred1 + offset)
        assert isinstance(mod1.predict(exog=exog1, exposure=pd.Series(exposure1)), np.ndarray)