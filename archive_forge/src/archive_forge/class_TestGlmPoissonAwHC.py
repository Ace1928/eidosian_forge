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
class TestGlmPoissonAwHC(CheckWeight):

    @classmethod
    def setup_class(cls):
        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs
        cls.corr_fact = np.sqrt((wsum - 1.0) / wsum) * 0.9851847359990561
        mod = GLM(cpunish_data.endog, cpunish_data.exog, family=sm.families.Poisson(), var_weights=aweights)
        cls.res1 = mod.fit(cov_type='HC0')
        cls.res2 = res_stata.results_poisson_aweight_hc1