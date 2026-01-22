import io
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose
from statsmodels.regression.linear_model import WLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.stats.meta_analysis import (
from .results import results_meta
class TestMetaBinOR:

    @classmethod
    def setup_class(cls):
        cls.res2 = res2 = results_meta.results_or_dl_hk
        cls.dta = (res2.event_e, res2.n_e, res2.event_c, res2.n_c)
        eff, var_eff = effectsize_2proportions(*cls.dta, statistic='or')
        res1 = combine_effects(eff, var_eff, method_re='chi2', use_t=True)
        cls.eff = eff
        cls.var_eff = var_eff
        cls.res1 = res1

    def test_basic(self):
        res1 = self.res1
        res2 = self.res2
        assert_allclose(self.eff, res2.TE, rtol=1e-13)
        assert_allclose(self.var_eff, res2.seTE ** 2, rtol=1e-13)
        assert_allclose(res1.mean_effect_fe, res2.TE_fixed, rtol=1e-13)
        assert_allclose(res1.sd_eff_w_fe, res2.seTE_fixed, rtol=1e-13)
        assert_allclose(res1.q, res2.Q, rtol=1e-13)
        assert_allclose(res1.tau2, res2.tau2, rtol=1e-10)
        assert_allclose(res1.mean_effect_re, res2.TE_random, rtol=1e-13)
        assert_allclose(res1.sd_eff_w_re_hksj, res2.seTE_random, rtol=1e-13)
        th = res1.test_homogeneity()
        q, pv = th
        df = th.df
        assert_allclose(q, res2.Q, rtol=1e-13)
        assert_allclose(pv, res2.pval_Q, rtol=1e-13)
        assert_allclose(df, res2.df_Q, rtol=1e-13)
        assert_allclose(res1.i2, res2.I2, rtol=1e-13)
        assert_allclose(res1.h2, res2.H ** 2, rtol=1e-13)
        ci = res1.conf_int(use_t=True)
        assert_allclose(ci[3][0], res2.lower_random, rtol=1e-13)
        assert_allclose(ci[3][1], res2.upper_random, rtol=1e-10)
        ci = res1.conf_int(use_t=False)
        assert_allclose(ci[0][0], res2.lower_fixed, rtol=1e-13)
        assert_allclose(ci[0][1], res2.upper_fixed, rtol=1e-13)
        weights = 1 / self.var_eff
        mod_glm = GLM(self.eff, np.ones(len(self.eff)), var_weights=weights)
        res_glm = mod_glm.fit()
        assert_allclose(res_glm.params, res2.TE_fixed, rtol=1e-13)
        weights = 1 / (self.var_eff + res1.tau2)
        mod_glm = GLM(self.eff, np.ones(len(self.eff)), var_weights=weights)
        res_glm = mod_glm.fit()
        assert_allclose(res_glm.params, res2.TE_random, rtol=1e-13)

    @pytest.mark.matplotlib
    def test_plot(self):
        res1 = self.res1
        res1.plot_forest(use_t=False)
        res1.plot_forest(use_exp=True, use_t=False)
        res1.plot_forest(alpha=0.01, use_t=False)
        with pytest.raises(TypeError, match='unexpected keyword'):
            res1.plot_forest(junk=5, use_t=False)