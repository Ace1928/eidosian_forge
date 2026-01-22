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
class TestPoissonCluGeneric(CheckCountRobustMixin):

    @classmethod
    def setup_class(cls):
        cls.res2 = results_st.results_poisson_clu
        mod = smd.Poisson(endog, exog)
        cls.res1 = res1 = mod.fit(disp=False)
        debug = False
        if debug:
            cls.bse_nonrobust = cls.res1.bse.copy()
            cls.res1 = res1 = mod.fit(disp=False)
            cls.get_robust_clu()
            cls.res3 = cls.res1
            cls.bse_rob3 = cls.bse_rob.copy()
            cls.res1 = res1 = mod.fit(disp=False)
        from statsmodels.base.covtype import get_robustcov_results
        get_robustcov_results(cls.res1._results, 'cluster', groups=group, use_correction=True, df_correction=True, use_t=False, use_self=True)
        cls.bse_rob = cls.res1.bse
        cls.corr_fact = cls.get_correction_factor(cls.res1)