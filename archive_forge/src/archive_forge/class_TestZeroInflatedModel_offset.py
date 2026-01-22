from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
class TestZeroInflatedModel_offset(CheckGeneric):

    @classmethod
    def setup_class(cls):
        data = sm.datasets.randhie.load()
        cls.endog = np.asarray(data.endog)
        data.exog = np.asarray(data.exog)
        exog = sm.add_constant(data.exog[:, 1:4], prepend=False)
        exog_infl = sm.add_constant(data.exog[:, 0], prepend=False)
        cls.res1 = sm.ZeroInflatedPoisson(data.endog, exog, exog_infl=exog_infl, offset=data.exog[:, 7]).fit(method='newton', maxiter=500, disp=False)
        cls.res1._results._attach_nullmodel = True
        cls.init_keys = ['exog_infl', 'exposure', 'inflation', 'offset']
        cls.init_kwds = {'inflation': 'logit'}
        res2 = RandHIE.zero_inflated_poisson_offset
        cls.res2 = res2

    def test_exposure(self):
        model1 = self.res1.model
        offset = model1.offset
        model3 = sm.ZeroInflatedPoisson(model1.endog, model1.exog, exog_infl=model1.exog_infl, exposure=np.exp(offset))
        res3 = model3.fit(start_params=self.res1.params, method='newton', maxiter=500, disp=False)
        assert_allclose(res3.params, self.res1.params, atol=1e-06, rtol=1e-06)
        fitted1 = self.res1.predict()
        fitted3 = res3.predict()
        assert_allclose(fitted3, fitted1, atol=1e-06, rtol=1e-06)
        ex = model1.exog
        ex_infl = model1.exog_infl
        offset = model1.offset
        fitted1_0 = self.res1.predict(exog=ex, exog_infl=ex_infl, offset=offset.tolist())
        fitted3_0 = res3.predict(exog=ex, exog_infl=ex_infl, exposure=np.exp(offset))
        assert_allclose(fitted3_0, fitted1_0, atol=1e-06, rtol=1e-06)
        ex = model1.exog[:10:2]
        ex_infl = model1.exog_infl[:10:2]
        offset = offset[:10:2]
        fitted1_2 = self.res1.predict(exog=ex, exog_infl=ex_infl, offset=offset)
        fitted3_2 = res3.predict(exog=ex, exog_infl=ex_infl, exposure=np.exp(offset))
        assert_allclose(fitted3_2, fitted1_2, atol=1e-06, rtol=1e-06)
        assert_allclose(fitted1_2, fitted1[:10:2], atol=1e-06, rtol=1e-06)
        assert_allclose(fitted3_2, fitted1[:10:2], atol=1e-06, rtol=1e-06)
        fitted1_3 = self.res1.predict(exog=ex, exog_infl=ex_infl)
        fitted3_3 = res3.predict(exog=ex, exog_infl=ex_infl)
        assert_allclose(fitted3_3, fitted1_3, atol=1e-06, rtol=1e-06)