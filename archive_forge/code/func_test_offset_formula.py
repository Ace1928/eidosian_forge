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
def test_offset_formula(self):
    n = 50
    np.random.seed(34234)
    X1 = np.random.normal(size=n)
    X2 = np.random.normal(size=n)
    groups = np.kron(np.arange(25), np.r_[1, 1])
    offset = np.random.uniform(1, 2, size=n)
    exposure = np.exp(offset)
    Y = np.random.poisson(0.1 * (X1 + X2) + 2 * offset, size=n)
    data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'groups': groups, 'offset': offset, 'exposure': exposure})
    fml = 'Y ~ X1 + X2'
    model1 = gee.GEE.from_formula(fml, groups, data, family=families.Poisson(), offset='offset')
    result1 = model1.fit()
    assert_equal(result1.converged, True)
    model2 = gee.GEE.from_formula(fml, groups, data, family=families.Poisson(), offset=offset)
    result2 = model2.fit(start_params=result1.params)
    assert_allclose(result1.params, result2.params)
    assert_equal(result2.converged, True)
    model3 = gee.GEE.from_formula(fml, groups, data, family=families.Poisson(), exposure=exposure)
    result3 = model3.fit(start_params=result1.params)
    assert_allclose(result1.params, result3.params)
    assert_equal(result3.converged, True)
    model4 = gee.GEE.from_formula(fml, groups, data, family=families.Poisson(), exposure='exposure')
    result4 = model4.fit(start_params=result1.params)
    assert_allclose(result1.params, result4.params)
    assert_equal(result4.converged, True)
    model5 = gee.GEE.from_formula(fml, groups, data, family=families.Poisson(), exposure='exposure', offset='offset')
    result5 = model5.fit()
    assert_equal(result5.converged, True)
    model6 = gee.GEE.from_formula(fml, groups, data, family=families.Poisson(), offset=2 * offset)
    result6 = model6.fit(start_params=result5.params)
    assert_allclose(result5.params, result6.params)
    assert_equal(result6.converged, True)