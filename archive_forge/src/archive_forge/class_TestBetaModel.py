import io
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import patsy
from statsmodels.api import families
from statsmodels.tools.sm_exceptions import (
from statsmodels.othermod.betareg import BetaModel
from .results import results_betareg as resultsb
class TestBetaModel:

    @classmethod
    def setup_class(cls):
        model = 'I(food/income) ~ income + persons'
        cls.income_fit = BetaModel.from_formula(model, income).fit()
        model = cls.model = 'methylation ~ gender + CpG'
        Z = cls.Z = patsy.dmatrix('~ age', methylation)
        mod = BetaModel.from_formula(model, methylation, exog_precision=Z, link_precision=links.Identity())
        cls.meth_fit = mod.fit()
        mod = BetaModel.from_formula(model, methylation, exog_precision=Z, link_precision=links.Log())
        cls.meth_log_fit = mod.fit()

    def test_income_coefficients(self):
        rslt = self.income_fit
        assert_close(rslt.params[:-1], expected_income_mean['Estimate'], 0.001)
        assert_close(rslt.tvalues[:-1], expected_income_mean['zvalue'], 0.1)
        assert_close(rslt.pvalues[:-1], expected_income_mean['Pr(>|z|)'], 0.001)

    def test_income_precision(self):
        rslt = self.income_fit
        assert_close(np.exp(rslt.params[-1:]), expected_income_precision['Estimate'], 0.001)
        assert_close(rslt.pvalues[-1:], expected_income_precision['Pr(>|z|)'], 0.001)

    def test_methylation_coefficients(self):
        rslt = self.meth_fit
        assert_close(rslt.params[:-2], expected_methylation_mean['Estimate'], 0.01)
        assert_close(rslt.tvalues[:-2], expected_methylation_mean['zvalue'], 0.1)
        assert_close(rslt.pvalues[:-2], expected_methylation_mean['Pr(>|z|)'], 0.01)

    def test_methylation_precision(self):
        rslt = self.meth_log_fit
        assert_allclose(rslt.params[-2:], expected_methylation_precision['Estimate'], atol=1e-05, rtol=1e-10)

    def test_precision_formula(self):
        m = BetaModel.from_formula(self.model, methylation, exog_precision_formula='~ age', link_precision=links.Identity())
        rslt = m.fit()
        assert_close(rslt.params, self.meth_fit.params, 1e-10)
        assert isinstance(rslt.params, pd.Series)
        with pytest.warns(ValueWarning, match='unknown kwargs'):
            BetaModel.from_formula(self.model, methylation, exog_precision_formula='~ age', link_precision=links.Identity(), junk=False)

    def test_scores(self):
        model, Z = (self.model, self.Z)
        for link in (links.Identity(), links.Log()):
            mod2 = BetaModel.from_formula(model, methylation, exog_precision=Z, link_precision=link)
            rslt_m = mod2.fit()
            analytical = rslt_m.model.score(rslt_m.params * 1.01)
            numerical = rslt_m.model._score_check(rslt_m.params * 1.01)
            assert_allclose(analytical, numerical, rtol=1e-06, atol=1e-06)
            assert_allclose(link.inverse(analytical[3:]), link.inverse(numerical[3:]), rtol=5e-07, atol=5e-06)

    def test_results_other(self):
        rslt = self.meth_fit
        distr = rslt.get_distribution()
        mean, var = distr.stats()
        assert_allclose(rslt.fittedvalues, mean, rtol=1e-13)
        assert_allclose(rslt.model._predict_var(rslt.params), var, rtol=1e-13)
        resid = rslt.model.endog - mean
        assert_allclose(rslt.resid, resid, rtol=1e-12)
        assert_allclose(rslt.resid_pearson, resid / np.sqrt(var), rtol=1e-12)