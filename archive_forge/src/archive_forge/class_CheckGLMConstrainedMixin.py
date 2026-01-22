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
class CheckGLMConstrainedMixin(CheckPoissonConstrainedMixin):

    def test_glm(self):
        res2 = self.res2
        res1 = self.res1m
        assert_allclose(res1.aic, res2.infocrit[4], rtol=1e-10)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            assert_allclose(res1.bic, res2.bic, rtol=1e-10)
        assert_allclose(res1.deviance, res2.deviance, rtol=1e-10)