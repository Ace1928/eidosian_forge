import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
import scipy.stats as stats
from statsmodels.discrete.discrete_model import Logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tools.tools import add_constant
from .results.results_ordinal_model import data_store as ds
class TestProbitModel(CheckOrdinalModelMixin):

    @classmethod
    def setup_class(cls):
        data = ds.df
        data_unordered = ds.df_unordered
        mod = OrderedModel(data['apply'].values.codes, np.asarray(data[['pared', 'public', 'gpa']], float), distr='probit')
        res = mod.fit(method='bfgs', disp=False)
        modp = OrderedModel(data['apply'], data[['pared', 'public', 'gpa']], distr='probit')
        resp = modp.fit(method='bfgs', disp=False)
        modf = OrderedModel.from_formula('apply ~ pared + public + gpa - 1', data={'apply': data['apply'].values.codes, 'pared': data['pared'], 'public': data['public'], 'gpa': data['gpa']}, distr='probit')
        resf = modf.fit(method='bfgs', disp=False)
        modu = OrderedModel(data_unordered['apply'].values.codes, np.asarray(data_unordered[['pared', 'public', 'gpa']], float), distr='probit')
        resu = modu.fit(method='bfgs', disp=False)
        from .results.results_ordinal_model import res_ord_probit as res2
        cls.res2 = res2
        cls.res1 = res
        cls.resp = resp
        cls.resf = resf
        cls.resu = resu
        cls.pred_table = np.array([[202, 18, 0, 220], [112, 28, 0, 140], [27, 13, 0, 40], [341, 59, 0, 400]], dtype=np.int64)

    def test_loglikerelated(self):
        res1 = self.res1
        mod = res1.model
        fact = 1.1
        score1 = mod.score(res1.params * fact)
        score_obs_numdiff = mod.score_obs(res1.params * fact)
        score_obs_exog = mod.score_obs_(res1.params * fact)
        assert_allclose(score_obs_numdiff.sum(0), score1, atol=1e-06)
        assert_allclose(score_obs_exog.sum(0), score1[:mod.k_vars], atol=1e-06)
        mod_null = OrderedModel(mod.endog, None, offset=np.zeros(mod.nobs), distr=mod.distr)
        null_params = mod.start_params
        res_null = mod_null.fit(method='bfgs', disp=False)
        assert_allclose(res_null.params, null_params[mod.k_vars:], rtol=1e-08)
        assert_allclose(res1.llnull, res_null.llf, rtol=1e-08)

    def test_formula_categorical(self):
        resp = self.resp
        data = ds.df
        formula = 'apply ~ pared + public + gpa - 1'
        modf2 = OrderedModel.from_formula(formula, data, distr='probit')
        resf2 = modf2.fit(method='bfgs', disp=False)
        assert_allclose(resf2.params, resp.params, atol=1e-08)
        assert modf2.exog_names == resp.model.exog_names
        assert modf2.data.ynames == resp.model.data.ynames
        assert hasattr(modf2.data, 'frame')
        assert not hasattr(modf2, 'frame')
        msg = 'Only ordered pandas Categorical'
        with pytest.raises(ValueError, match=msg):
            OrderedModel.from_formula('apply ~ pared + public + gpa - 1', data={'apply': np.asarray(data['apply']), 'pared': data['pared'], 'public': data['public'], 'gpa': data['gpa']}, distr='probit')

    def test_offset(self):
        resp = self.resp
        data = ds.df
        offset = np.ones(len(data))
        formula = 'apply ~ pared + public + gpa - 1'
        modf2 = OrderedModel.from_formula(formula, data, offset=offset, distr='probit')
        resf2 = modf2.fit(method='bfgs', disp=False)
        resf2_params = np.asarray(resf2.params)
        resp_params = np.asarray(resp.params)
        assert_allclose(resf2_params[:3], resp_params[:3], atol=0.0002)
        assert_allclose(resf2_params[3], resp_params[3] + 1, atol=0.0002)
        fitted = resp.predict()
        fitted2 = resf2.predict()
        assert_allclose(fitted2, fitted, atol=0.0002)
        pred_ones = resf2.predict(data[:6], offset=np.ones(6))
        assert_allclose(pred_ones, fitted[:6], atol=0.0002)
        pred_zero1 = resf2.predict(data[:6])
        pred_zero2 = resf2.predict(data[:6], offset=0)
        assert_allclose(pred_zero1, pred_zero2, atol=0.0002)
        pred_zero = resp.predict(data[['pared', 'public', 'gpa']].iloc[:6], offset=-np.ones(6))
        assert_allclose(pred_zero1, pred_zero, atol=0.0002)
        params_adj = resp.params.copy()
        params_adj.iloc[3] += 1
        fitted_zero = resp.model.predict(params_adj)
        assert_allclose(pred_zero1, fitted_zero[:6], atol=0.0002)