import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels import datasets
from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.tools.sm_exceptions import (
from statsmodels.distributions.discrete import (
from statsmodels.discrete.truncated_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results.results_discrete import RandHIE
from .results import results_truncated as results_t
from .results import results_truncated_st as results_ts
class CheckHurdlePredict:

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert res1.df_model == res2.df_model
        assert res1.df_resid == res2.df_resid
        assert res1.model.k_extra == res2.k_extra
        assert len(res1.model.exog_names) == res2.k_params
        assert res1.model.exog_names == res2.exog_names
        res1.summary()

    def test_predict(self):
        res1 = self.res1
        endog = res1.model.endog
        exog = res1.model.exog
        pred_mean = res1.predict(which='mean').mean()
        assert_allclose(pred_mean, endog.mean(), rtol=0.01)
        mask_nz = endog > 0
        mean_nz = endog[mask_nz].mean()
        pred_mean_nz = res1.predict(which='mean-nonzero').mean()
        assert_allclose(pred_mean_nz, mean_nz, rtol=0.05)
        pred_mean_nnz = res1.predict(exog=exog[mask_nz], which='mean-nonzero').mean()
        assert_allclose(pred_mean_nnz, mean_nz, rtol=0.0005)
        pred_mean_nzm = res1.results_count.predict(which='mean').mean()
        assert_allclose(pred_mean_nzm, mean_nz, rtol=0.0005)
        assert_allclose(pred_mean_nzm, pred_mean_nnz, rtol=0.0001)
        pred_var = res1.predict(which='var').mean()
        assert_allclose(pred_var, res1.resid.var(), rtol=0.05)
        pred_var = res1.results_count.predict(which='var').mean()
        assert_allclose(pred_var, res1.resid[endog > 0].var(), rtol=0.05)
        freq = np.bincount(endog.astype(int)) / len(endog)
        pred_prob = res1.predict(which='prob').mean(0)
        assert_allclose(pred_prob, freq, rtol=0.005, atol=0.01)
        dia_hnb = res1.get_diagnostic()
        assert_allclose(dia_hnb.probs_predicted.mean(0), pred_prob, rtol=1e-10)
        try:
            dia_hnb.plot_probs()
        except ImportError:
            pass
        pred_prob0 = res1.predict(which='prob-zero').mean(0)
        assert_allclose(pred_prob0, freq[0], rtol=0.0001)
        assert_allclose(pred_prob0, pred_prob[0], rtol=1e-10)