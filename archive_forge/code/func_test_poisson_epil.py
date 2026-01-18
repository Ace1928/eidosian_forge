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
def test_poisson_epil(self):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(cur_dir, 'results', 'epil.csv')
    data = pd.read_csv(fname)
    fam = families.Poisson()
    ind = cov_struct.Independence()
    mod1 = gee.GEE.from_formula('y ~ age + trt + base', data['subject'], data, cov_struct=ind, family=fam)
    rslt1 = mod1.fit(cov_type='naive')
    from statsmodels.genmod.generalized_linear_model import GLM
    mod2 = GLM.from_formula('y ~ age + trt + base', data, family=families.Poisson())
    rslt2 = mod2.fit()
    rslt1 = rslt1._results
    rslt2 = rslt2._results
    assert_allclose(rslt1.params, rslt2.params, rtol=1e-06, atol=1e-06)
    assert_allclose(rslt1.bse, rslt2.bse, rtol=1e-06, atol=1e-06)