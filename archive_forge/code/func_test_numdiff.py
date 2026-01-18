import warnings
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.discrete.discrete_model import Poisson, Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import family
from statsmodels.sandbox.regression.penalized import TheilGLS
from statsmodels.base._penalized import PenalizedMixin
import statsmodels.base._penalties as smpen
def test_numdiff(self):
    res1 = self.res1
    p = res1.params * 0.98
    kwds = {'scale': 1} if isinstance(res1.model, GLM) else {}
    assert_allclose(res1.model.score(p, **kwds)[self.exog_index], res1.model.score_numdiff(p, **kwds)[self.exog_index], rtol=0.025)
    if not self.skip_hessian:
        if isinstance(self.exog_index, slice):
            idx1 = idx2 = self.exog_index
        else:
            idx1 = self.exog_index[:, None]
            idx2 = self.exog_index
        h1 = res1.model.hessian(res1.params, **kwds)[idx1, idx2]
        h2 = res1.model.hessian_numdiff(res1.params, **kwds)[idx1, idx2]
        assert_allclose(h1, h2, rtol=0.02)