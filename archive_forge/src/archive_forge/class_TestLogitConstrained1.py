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
class TestLogitConstrained1(CheckGLMConstrainedMixin):

    @classmethod
    def setup_class(cls):
        cls.idx = slice(None)
        cls.res2 = reslogit.results_constraint1
        mod1 = Logit(spector_data.endog, spector_data.exog)
        constr = 'x1 = 2.8'
        cls.res1m = mod1.fit_constrained(constr, method='bfgs')
        R, q = (cls.res1m.constraints.coefs, cls.res1m.constraints.constants)
        cls.res1 = fit_constrained(mod1, R, q, fit_kwds={'method': 'bfgs'})

    @pytest.mark.skip(reason='not a GLM')
    def test_glm(self):
        return