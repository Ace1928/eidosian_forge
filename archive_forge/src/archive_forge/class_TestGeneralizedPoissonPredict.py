import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.tools.tools import add_constant
from statsmodels.base._prediction_inference import PredictionResultsMonotonic
from statsmodels.discrete.discrete_model import (
from statsmodels.discrete.count_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp
class TestGeneralizedPoissonPredict(CheckExtras):

    @classmethod
    def setup_class(cls):
        cls.klass = GeneralizedPoisson
        mod1 = GeneralizedPoisson(endog, exog)
        res1 = mod1.fit(method='newton')
        cls.res1 = res1
        cls.res2 = resp.results_nb_docvis
        cls.pred_kwds_mean = {}
        cls.pred_kwds_6 = {}
        cls.k_infl = 0
        cls.rtol = 1e-08