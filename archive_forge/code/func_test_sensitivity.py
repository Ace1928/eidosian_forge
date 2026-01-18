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
def test_sensitivity(self):
    va = cov_struct.Exchangeable()
    family = families.Gaussian()
    np.random.seed(34234)
    n = 100
    Y = np.random.normal(size=n)
    X1 = np.random.normal(size=n)
    X2 = np.random.normal(size=n)
    groups = np.kron(np.arange(50), np.r_[1, 1])
    D = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2})
    mod = gee.GEE.from_formula('Y ~ X1 + X2', groups, D, family=family, cov_struct=va)
    rslt = mod.fit()
    ps = rslt.params_sensitivity(0, 0.5, 2)
    assert_almost_equal(len(ps), 2)
    assert_almost_equal([x.cov_struct.dep_params for x in ps], [0.0, 0.5])
    assert_almost_equal([np.asarray(x.params)[0] for x in ps], [0.1696214707458818, 0.17836097387799127])