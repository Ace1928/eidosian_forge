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
def test_unchanging_degrees_of_freedom():
    data = load_randhie()
    warnings.simplefilter('error')
    model = sm.NegativeBinomial(data.endog, data.exog, loglike_method='nb2')
    params = np.array([-0.05654134, -0.21213734, 0.08783102, -0.02991825, 0.22902315, 0.06210253, 0.06799444, 0.08406794, 0.18530092, 1.36645186])
    res1 = model.fit(start_params=params, disp=0)
    assert_equal(res1.df_model, 8)
    reg_params = np.array([-0.04854, -0.15019404, 0.08363671, -0.03032834, 0.17592454, 0.06440753, 0.01584555, 0.0, 0.0, 1.36984628])
    res2 = model.fit_regularized(alpha=100, start_params=reg_params, disp=0)
    assert_(res2.df_model != 8)
    res3 = model.fit(start_params=params, disp=0)
    assert_equal(res3.df_model, res1.df_model)
    assert_equal(res3.df_resid, res1.df_resid)