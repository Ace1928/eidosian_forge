import os
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pandas as pd
import pytest
import patsy
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen
from statsmodels.gam.smooth_basis import (BSplines, CyclicCubicSplines)
from statsmodels.gam.generalized_additive_model import (
from statsmodels.tools.linalg import matrix_sqrt, transf_constraints
from .results import results_pls, results_mpg_bs, results_mpg_bs_poisson
class TestGAMMPG:

    @classmethod
    def setup_class(cls):
        sp = np.array([6.46225497484073, 0.81532465890585])
        s_scale = np.array([2.95973613706629e-07, 0.000126203730141359])
        x_spline = df_autos[['weight', 'hp']].values
        exog = patsy.dmatrix('fuel + drive', data=df_autos)
        cc = CyclicCubicSplines(x_spline, df=[6, 5], constraints='center')
        gam_cc = GLMGam(df_autos['city_mpg'], exog=exog, smoother=cc, alpha=(1 / s_scale * sp / 2).tolist())
        cls.res1a = gam_cc.fit()
        gam_cc = GLMGam(df_autos['city_mpg'], exog=exog, smoother=cc, alpha=(1 / s_scale * sp / 2).tolist())
        cls.res1b = gam_cc.fit(method='newton')

    def test_exog(self):
        file_path = os.path.join(cur_dir, 'results', 'autos_exog.csv')
        df_exog = pd.read_csv(file_path)
        res2_exog = df_exog.values
        for res1 in [self.res1a, self.res1b]:
            exog = res1.model.exog
            assert_allclose(exog, res2_exog, atol=1e-14)

    def test_fitted(self):
        file_path = os.path.join(cur_dir, 'results', 'autos_predict.csv')
        df_pred = pd.read_csv(file_path, index_col='Row.names')
        df_pred.index = df_pred.index - 1
        res2_fittedvalues = df_pred['fit'].values
        res2_se_mean = df_pred['se_fit'].values
        for res1 in [self.res1a, self.res1b]:
            pred = res1.get_prediction()
            self.rtol_fitted = 1e-05
            assert_allclose(res1.fittedvalues, res2_fittedvalues, rtol=1e-10)
            assert_allclose(pred.predicted_mean, res2_fittedvalues, rtol=1e-10)
            corr_fact = 1
            assert_allclose(pred.se_mean, res2_se_mean * corr_fact, rtol=1e-10)