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
class TestGlmPoissonFwClu(CheckWeight):

    @classmethod
    def setup_class(cls):
        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs
        gid = np.arange(1, 17 + 1) // 2
        n_groups = len(np.unique(gid))
        cls.corr_fact = 1 / np.sqrt(n_groups / (n_groups - 1))
        cov_kwds = {'groups': gid, 'use_correction': False}
        with pytest.warns(SpecificationWarning):
            mod = GLM(cpunish_data.endog, cpunish_data.exog, family=sm.families.Poisson(), freq_weights=fweights)
            cls.res1 = mod.fit(cov_type='cluster', cov_kwds=cov_kwds)
        cls.res2 = res_stata.results_poisson_fweight_clu1