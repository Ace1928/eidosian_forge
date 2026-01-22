from io import StringIO
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import patsy
import pytest
from statsmodels import datasets
from statsmodels.base._constraints import fit_constrained
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant
from .results import (
class TestLogitConstrained2HC(CheckGLMConstrainedMixin):

    @classmethod
    def setup_class(cls):
        cls.idx = slice(None)
        cls.res2 = reslogit.results_constraint2_robust
        mod1 = Logit(spector_data.endog, spector_data.exog)
        cov_type = 'HC0'
        cov_kwds = {'scaling_factor': 32 / 31}
        constr = 'x1 - x3 = 0'
        cls.res1m = mod1.fit_constrained(constr, cov_type=cov_type, cov_kwds=cov_kwds, tol=1e-10)
        R, q = (cls.res1m.constraints.coefs, cls.res1m.constraints.constants)
        cls.res1 = fit_constrained(mod1, R, q, fit_kwds={'tol': 1e-10, 'cov_type': cov_type, 'cov_kwds': cov_kwds})
        cls.constraints_rq = (R, q)

    @pytest.mark.skip(reason='not a GLM')
    def test_glm(self):
        return