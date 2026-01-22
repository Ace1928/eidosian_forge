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
class TestTruncatedLFPoisson1St(CheckTruncatedST):

    @classmethod
    def setup_class(cls):
        endog = DATA['docvis']
        exog_names = ['aget', 'totchr', 'const']
        exog = DATA[exog_names]
        cls.res1 = TruncatedLFPoisson(endog, exog, truncation=1).fit(method='bfgs', maxiter=300)
        cls.res2 = results_ts.results_trunc_poisson1