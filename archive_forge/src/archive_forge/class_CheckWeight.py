import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets.cpunish import load
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.tools import add_constant
from .results import (
class CheckWeight:

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(res1.params, res2.params, atol=1e-06, rtol=2e-06)
        corr_fact = getattr(self, 'corr_fact', 1)
        if hasattr(res2, 'normalized_cov_params'):
            assert_allclose(res1.normalized_cov_params, res2.normalized_cov_params, atol=1e-08, rtol=2e-06)
        if isinstance(self, (TestRepeatedvsAggregated, TestRepeatedvsAverage, TestTweedieRepeatedvsAggregated, TestTweedieRepeatedvsAverage, TestBinomial0RepeatedvsAverage, TestBinomial0RepeatedvsDuplicated)):
            return None
        assert_allclose(res1.bse, corr_fact * res2.bse, atol=1e-06, rtol=2e-06)
        if isinstance(self, TestBinomialVsVarWeights):
            return None
        if isinstance(self, TestGlmGaussianWLS):
            return None
        if not isinstance(self, (TestGlmGaussianAwNr, TestGlmGammaAwNr)):
            assert_allclose(res1.llf, res2.ll, atol=1e-06, rtol=1e-07)
        assert_allclose(res1.deviance, res2.deviance, atol=1e-06, rtol=1e-07)

    def test_residuals(self):
        if isinstance(self, (TestRepeatedvsAggregated, TestRepeatedvsAverage, TestTweedieRepeatedvsAggregated, TestTweedieRepeatedvsAverage, TestBinomial0RepeatedvsAverage, TestBinomial0RepeatedvsDuplicated)):
            return None
        res1 = self.res1
        res2 = self.res2
        if not hasattr(res2, 'resids'):
            return None
        resid_all = dict(zip(res2.resids_colnames, res2.resids.T))
        assert_allclose(res1.resid_response, resid_all['resid_response'], atol=1e-06, rtol=2e-06)
        assert_allclose(res1.resid_pearson, resid_all['resid_pearson'], atol=1e-06, rtol=2e-06)
        assert_allclose(res1.resid_deviance, resid_all['resid_deviance'], atol=1e-06, rtol=2e-06)
        assert_allclose(res1.resid_working, resid_all['resid_working'], atol=1e-06, rtol=2e-06)
        if resid_all.get('resid_anscombe') is None:
            return None
        resid_a = res1.resid_anscombe
        resid_a1 = resid_all['resid_anscombe'] * np.sqrt(res1._var_weights)
        assert_allclose(resid_a, resid_a1, atol=1e-06, rtol=2e-06)

    def test_compare_optimizers(self):
        res1 = self.res1
        if isinstance(res1.model.family, sm.families.Tweedie):
            method = 'newton'
            optim_hessian = 'eim'
        else:
            method = 'bfgs'
            optim_hessian = 'oim'
        if isinstance(self, (TestGlmPoissonFwHC, TestGlmPoissonAwHC, TestGlmPoissonFwClu, TestBinomial0RepeatedvsAverage)):
            return None
        start_params = res1.params
        res2 = self.res1.model.fit(start_params=start_params, method=method, optim_hessian=optim_hessian)
        assert_allclose(res1.params, res2.params, atol=0.001, rtol=0.002)
        H = res2.model.hessian(res2.params, observed=False)
        res2_bse = np.sqrt(-np.diag(np.linalg.inv(H)))
        assert_allclose(res1.bse, res2_bse, atol=0.001, rtol=0.001)

    def test_pearson_chi2(self):
        if hasattr(self.res2, 'chi2'):
            assert_allclose(self.res1.pearson_chi2, self.res2.deviance_p, atol=1e-06, rtol=1e-06)

    def test_getprediction(self):
        pred = self.res1.get_prediction()
        assert_allclose(pred.linpred.se_mean, pred.linpred.se_mean, rtol=1e-10)