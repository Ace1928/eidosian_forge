import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels import datasets
from statsmodels.tools.tools import add_constant
from statsmodels.tools.testing import Holder
from statsmodels.tools.sm_exceptions import (
from statsmodels.distributions.discrete import (
from statsmodels.discrete.truncated_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results.results_discrete import RandHIE
from .results import results_truncated as results_t
from .results import results_truncated_st as results_ts
class TestTruncatedLFPoissonModel(CheckResults):

    @classmethod
    def setup_class(cls):
        data = datasets.randhie.load()
        exog = add_constant(np.asarray(data.exog)[:, :4], prepend=False)
        mod = TruncatedLFPoisson(data.endog, exog, truncation=5)
        cls.res1 = mod.fit(method='newton', maxiter=500)
        res2 = RandHIE()
        res2.truncated_poisson()
        cls.res2 = res2