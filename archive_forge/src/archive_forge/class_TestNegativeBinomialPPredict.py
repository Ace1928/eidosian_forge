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
class TestNegativeBinomialPPredict(CheckPredict, CheckExtras):

    @classmethod
    def setup_class(cls):
        cls.klass = NegativeBinomialP
        res1 = NegativeBinomialP(endog, exog).fit(method='newton', maxiter=300)
        cls.res1 = res1
        cls.res2 = resp.results_nb_docvis
        cls.pred_kwds_mean = {}
        cls.pred_kwds_6 = {}
        cls.k_infl = 0
        cls.rtol = 1e-08