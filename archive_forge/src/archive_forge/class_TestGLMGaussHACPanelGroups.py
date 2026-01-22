import os
import numpy as np
import pandas as pd
import pytest
import statsmodels.discrete.discrete_model as smd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.regression.linear_model import OLS
from statsmodels.base.covtype import get_robustcov_results
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant
from numpy.testing import assert_allclose, assert_equal, assert_
import statsmodels.tools._testing as smt
from .results import results_count_robust_cluster as results_st
class TestGLMGaussHACPanelGroups(CheckDiscreteGLM):

    @classmethod
    def setup_class(cls):
        cls.cov_type = 'hac-panel'
        groups = np.repeat(np.arange(5), 7)[:-1]
        mod1 = GLM(endog.copy(), exog.copy(), family=families.Gaussian())
        kwds = dict(groups=pd.Series(groups), maxlags=2, kernel=sw.weights_uniform, use_correction='hac', df_correction=False)
        cls.res1 = mod1.fit(cov_type='hac-panel', cov_kwds=kwds)
        mod2 = OLS(endog, exog)
        cls.res2 = mod2.fit(cov_type='hac-panel', cov_kwds=kwds)