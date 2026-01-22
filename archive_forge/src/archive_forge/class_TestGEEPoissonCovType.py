from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
class TestGEEPoissonCovType(CheckConsistency):

    @classmethod
    def setup_class(cls):
        endog, exog, group_n = load_data('gee_poisson_1.csv')
        family = families.Poisson()
        vi = cov_struct.Independence()
        cls.mod = gee.GEE(endog, exog, group_n, None, family, vi)
        cls.start_params = np.array([-0.03644504, -0.05432094, 0.01566427, 0.57628591, -0.0046566, -0.47709315])

    def test_wrapper(self):
        endog, exog, group_n = load_data('gee_poisson_1.csv', icept=False)
        endog = pd.Series(endog)
        exog = pd.DataFrame(exog)
        group_n = pd.Series(group_n)
        family = families.Poisson()
        vi = cov_struct.Independence()
        mod = gee.GEE(endog, exog, group_n, None, family, vi)
        rslt2 = mod.fit()
        check_wrapper(rslt2)