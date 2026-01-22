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
class TestTheilPLS5(CheckGAMMixin):
    cov_type = 'data-prior'

    @classmethod
    def setup_class(cls):
        exog, penalty_matrix, restriction = cls._init()
        endog = data_mcycle['accel']
        modp = TheilGLS(endog, exog, r_matrix=restriction)
        s_scale_r = 0.02630734
        sigma_e = 1405.7950179165323
        cls.pw = pw = 1 / sigma_e / s_scale_r
        cls.res1 = modp.fit(pen_weight=pw, cov_type=cls.cov_type)
        cls.res2 = results_pls.pls5
        cls.rtol_fitted = 1e-07
        cls.covp_corrfact = 0.997869328442032

    def test_cov_robust(self):
        res1 = self.res1
        res2 = self.res2
        pw = res1.penalization_factor
        res1 = res1.model.fit(pen_weight=pw, cov_type='sandwich')
        assert_allclose(np.asarray(res1.cov_params()), res2.Ve * self.covp_corrfact, rtol=0.0001)

    def test_null_smoke(self):
        pytest.skip('llnull not available')