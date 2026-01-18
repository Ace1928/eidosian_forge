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
def test_start_null(self):
    endog, exog = (self.model.endog, self.model.exog)
    model_nb2 = NegativeBinomial(endog, exog, loglike_method='nb1')
    sp1 = model_nb2._get_start_params_null()
    sp0 = self.model._get_start_params_null()
    assert_allclose(sp0, sp1, rtol=1e-12)