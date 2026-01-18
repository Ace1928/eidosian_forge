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
def test_predict_exposure(self):
    n = 50
    np.random.seed(34234)
    X1 = np.random.normal(size=n)
    X2 = np.random.normal(size=n)
    groups = np.kron(np.arange(25), np.r_[1, 1])
    offset = np.random.uniform(1, 2, size=n)
    exposure = np.random.uniform(1, 2, size=n)
    Y = np.random.poisson(0.1 * (X1 + X2) + offset + np.log(exposure), size=n)
    data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'groups': groups, 'offset': offset, 'exposure': exposure})
    fml = 'Y ~ X1 + X2'
    model = gee.GEE.from_formula(fml, groups, data, family=families.Poisson(), offset='offset', exposure='exposure')
    result = model.fit()
    assert_equal(result.converged, True)
    pred1 = result.predict()
    pred2 = result.predict(offset=data['offset'])
    pred3 = result.predict(exposure=data['exposure'])
    pred4 = result.predict(offset=data['offset'], exposure=data['exposure'])
    pred5 = result.predict(exog=data[-10:], offset=data['offset'][-10:], exposure=data['exposure'][-10:])
    pred6 = result.predict(exog=result.model.exog[-10:], offset=data['offset'][-10:], exposure=data['exposure'][-10:], transform=False)
    assert_allclose(pred1, pred2)
    assert_allclose(pred1, pred3)
    assert_allclose(pred1, pred4)
    assert_allclose(pred1[-10:], pred5)
    assert_allclose(pred1[-10:], pred6)