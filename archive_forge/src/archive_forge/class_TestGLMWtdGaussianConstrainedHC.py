import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.genmod.families import family
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.tools import add_constant
class TestGLMWtdGaussianConstrainedHC(ConstrainedCompareWtdMixin):

    @classmethod
    def init(cls):
        cov_type = 'HC0'
        cls.res2 = cls.mod2.fit(cov_type=cov_type)
        mod = GLM(cls.endog, cls.exog, var_weights=cls.aweights)
        mod.exog_names[:] = ['const', 'x1', 'x2', 'x3', 'x4']
        cls.res1 = mod.fit_constrained('x1=0.5', cov_type=cov_type)