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
class TestGAM6Bfgs:

    @classmethod
    def setup_class(cls):
        s_scale = 0.0263073404164214
        cc = CyclicCubicSplines(data_mcycle['times'].values, df=[6])
        gam_cc = GLMGam(data_mcycle['accel'], smoother=cc, alpha=1 / s_scale / 2)
        cls.res1 = gam_cc.fit(method='bfgs')

    def test_fitted(self):
        res1 = self.res1
        pred = res1.get_prediction()
        self.rtol_fitted = 1e-05
        pls6_fittedvalues = np.array([2.45008146537851, 3.14145063965465, 5.24130119353225, 6.63476330674223, 7.99704341866374, 13.9351103077006, 14.5508371638833, 14.785647621276, 15.1176070735895, 14.8053514054347, 13.790412967255, 13.790412967255, 11.2997845518655, 9.51681958051473, 8.4811626302547])
        assert_allclose(res1.fittedvalues[:15], pls6_fittedvalues, rtol=self.rtol_fitted)
        assert_allclose(pred.predicted_mean[:15], pls6_fittedvalues, rtol=self.rtol_fitted)