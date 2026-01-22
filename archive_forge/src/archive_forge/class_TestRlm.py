import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
class TestRlm(CheckRlmResultsMixin):

    @classmethod
    def setup_class(cls):
        cls.data = load_stackloss()
        cls.data.exog = sm.add_constant(cls.data.exog, prepend=False)
        cls.decimal_standarderrors = DECIMAL_1
        cls.decimal_scale = DECIMAL_3
        model = RLM(cls.data.endog, cls.data.exog, M=norms.HuberT())
        cls.model = model
        results = model.fit()
        h2 = model.fit(cov='H2').bcov_scaled
        h3 = model.fit(cov='H3').bcov_scaled
        cls.res1 = results
        cls.res1.h2 = h2
        cls.res1.h3 = h3

    def setup_method(self):
        from .results.results_rlm import Huber
        self.res2 = Huber()

    @pytest.mark.smoke
    def test_summary(self):
        self.res1.summary()

    @pytest.mark.smoke
    def test_summary2(self):
        self.res1.summary2()

    @pytest.mark.smoke
    def test_chisq(self):
        assert isinstance(self.res1.chisq, np.ndarray)

    @pytest.mark.smoke
    def test_predict(self):
        assert isinstance(self.model.predict(self.res1.params), np.ndarray)