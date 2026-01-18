import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
def test_set_imputer(self):
    from statsmodels.regression.linear_model import RegressionResultsWrapper
    from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
    df = gendat()
    orig = df.copy()
    mx = pd.notnull(df)
    nrow, ncol = df.shape
    imp_data = mice.MICEData(df)
    imp_data.set_imputer('x1', 'x3 + x4 + x3*x4')
    imp_data.set_imputer('x2', 'x4 + I(x5**2)')
    imp_data.set_imputer('x3', model_class=sm.GLM, init_kwds={'family': sm.families.Binomial()})
    imp_data.update_all()
    assert_equal(imp_data.data.shape[0], nrow)
    assert_equal(imp_data.data.shape[1], ncol)
    assert_allclose(orig[mx], imp_data.data[mx])
    for j in range(1, 6):
        if j == 3:
            assert_equal(isinstance(imp_data.models['x3'], sm.GLM), True)
            assert_equal(isinstance(imp_data.models['x3'].family, sm.families.Binomial), True)
            assert_equal(isinstance(imp_data.results['x3'], GLMResultsWrapper), True)
        else:
            assert_equal(isinstance(imp_data.models['x%d' % j], sm.OLS), True)
            assert_equal(isinstance(imp_data.results['x%d' % j], RegressionResultsWrapper), True)
    fml = 'x1 ~ x3 + x4 + x3*x4'
    assert_equal(imp_data.conditional_formula['x1'], fml)
    fml = 'x4 ~ x1 + x2 + x3 + x5 + y'
    assert_equal(imp_data.conditional_formula['x4'], fml)
    assert tuple(imp_data._cycle_order) in (('x5', 'x3', 'x4', 'y', 'x2', 'x1'), ('x5', 'x4', 'x3', 'y', 'x2', 'x1'))