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
class TestGlmPoissonPwNr(CheckWeight):

    @classmethod
    def setup_class(cls):
        fweights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
        fweights = np.array(fweights)
        wsum = fweights.sum()
        nobs = len(cpunish_data.endog)
        aweights = fweights / wsum * nobs
        cls.res1 = GLM(cpunish_data.endog, cpunish_data.exog, family=sm.families.Poisson(), freq_weights=fweights).fit(cov_type='HC1')
        cls.res2 = res_stata.results_poisson_pweight_nonrobust

    @pytest.mark.xfail(reason='Known to fail', strict=True)
    def test_basic(self):
        super().test_basic()

    @pytest.mark.xfail(reason='Known to fail', strict=True)
    def test_compare_optimizers(self):
        super().test_compare_optimizers()