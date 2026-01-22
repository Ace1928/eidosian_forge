from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
class TestNegativeBinomialNB2Null(CheckNull):

    @classmethod
    def setup_class(cls):
        endog, exog = cls._get_data()
        cls.model = NegativeBinomial(endog, exog, loglike_method='nb2')
        cls.model_null = NegativeBinomial(endog, exog[:, 0], loglike_method='nb2')
        cls.res_null = cls.model_null.fit(start_params=[8, 0.5], method='bfgs', gtol=1e-06, maxiter=300, disp=0)
        cls.start_params = np.array([8.07216448, 0.01087238, 0.44024134])