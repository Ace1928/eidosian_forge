import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets.cpunish import load
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.tools import add_constant
from .results import (
def test_r_llf(self):
    res1 = self.res1
    res2 = self.res2
    model = self.res1.model
    scale = res1.scale * model.df_resid / model.wnobs
    wts = model.freq_weights
    llf = model.family.loglike(model.endog, res1.mu, freq_weights=wts, scale=scale)
    adj_sm = -1 / 2 * ((model.endog - res1.mu) ** 2).sum() / scale
    adj_r = -model.wnobs / 2 + np.sum(np.log(model.var_weights)) / 2
    llf_adj = llf - adj_sm + adj_r
    assert_allclose(llf_adj, res2.ll, atol=1e-06, rtol=1e-07)