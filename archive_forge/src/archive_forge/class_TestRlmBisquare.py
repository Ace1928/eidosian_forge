import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
class TestRlmBisquare(TestRlm):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.decimal_standarderrors = DECIMAL_1
        model = RLM(cls.data.endog, cls.data.exog, M=norms.TukeyBiweight())
        results = model.fit()
        h2 = model.fit(cov='H2').bcov_scaled
        h3 = model.fit(cov='H3').bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import BiSquare
        self.res2 = BiSquare()