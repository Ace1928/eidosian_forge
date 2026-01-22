import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.tools.tools import add_constant
from statsmodels.base._prediction_inference import PredictionResultsMonotonic
from statsmodels.discrete.discrete_model import (
from statsmodels.discrete.count_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp
class CheckExtras:

    def test_predict_linear(self):
        res1 = self.res1
        ex = np.asarray(exog[:5])
        pred = res1.get_prediction(ex, which='linear', **self.pred_kwds_mean)
        k_extra = len(res1.params) - ex.shape[1]
        if k_extra > 0:
            ex = np.column_stack((ex, np.zeros((ex.shape[0], k_extra))))
        tt = res1.t_test(ex)
        cip = pred.conf_int()
        cit = tt.conf_int()
        assert_allclose(cip, cit, rtol=1e-12)

    def test_score_test(self):
        res1 = self.res1
        modr = self.klass(endog, exog.values[:, :-1])
        resr = modr.fit(method='newton', maxiter=300)
        params_restr = np.concatenate([resr.params[:-1], [0], resr.params[-1:]])
        r_matrix = np.zeros((1, len(params_restr)))
        r_matrix[0, -2] = 1
        exog_extra = res1.model.exog[:, -1:]
        from statsmodels.base._parameter_inference import score_test
        sc1 = score_test(res1, params_constrained=params_restr, k_constraints=1)
        sc2 = score_test(resr, exog_extra=(exog_extra, None))
        assert_allclose(sc2[:2], sc1[:2])
        sc1_hc = score_test(res1, params_constrained=params_restr, k_constraints=1, r_matrix=r_matrix, cov_type='HC0')
        sc2_hc = score_test(resr, exog_extra=(exog_extra, None), cov_type='HC0')
        assert_allclose(sc2_hc[:2], sc1_hc[:2])

    def test_score_test_alpha(self):
        modr = self.klass(endog, exog.values[:, :-1])
        resr = modr.fit(method='newton', maxiter=300)
        params_restr = np.concatenate([resr.params[:], [0]])
        r_matrix = np.zeros((1, len(params_restr)))
        r_matrix[0, -1] = 1
        np.random.seed(987125643)
        exog_extra = 0.01 * np.random.randn(endog.shape[0])
        from statsmodels.base._parameter_inference import score_test, _scorehess_extra
        sh = _scorehess_extra(resr, exog_extra=None, exog2_extra=exog_extra, hess_kwds=None)
        assert not np.isnan(sh[0]).any()
        sc2 = score_test(resr, exog_extra=(None, exog_extra))
        assert sc2[1] > 0.01
        sc2_hc = score_test(resr, exog_extra=(None, exog_extra), cov_type='HC0')
        assert sc2_hc[1] > 0.01

    def test_influence(self):
        res1 = self.res1
        from statsmodels.stats.outliers_influence import MLEInfluence
        influ = MLEInfluence(res1)
        attrs = ['cooks_distance', 'd_fittedvalues', 'd_fittedvalues_scaled', 'd_params', 'dfbetas', 'hat_matrix_diag', 'resid_studentized']
        for attr in attrs:
            getattr(influ, attr)
        influ.summary_frame()