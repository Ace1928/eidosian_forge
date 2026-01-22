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
@classmethod
def _generate_endog(cls, linpred):
    sig_e = np.sqrt(0.1)
    np.random.seed(999)
    y = linpred + sig_e * np.random.rand(len(linpred))
    return y