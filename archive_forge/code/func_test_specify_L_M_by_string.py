import numpy as np
import pandas as pd
from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
from numpy.testing import assert_array_almost_equal, assert_raises
import patsy
def test_specify_L_M_by_string():
    mod = _MultivariateOLS.from_formula('Histamine0 + Histamine1 + Histamine3 + Histamine5 ~ Drug * Depleted', data)
    r = mod.fit()
    r1 = r.mv_test(hypotheses=[['Intercept', ['Intercept'], None]])
    a = [[0.026860766, 4, 6, 54.3435304, 7.5958561e-05], [0.973139234, 4, 6, 54.3435304, 7.5958561e-05], [36.2290202, 4, 6, 54.3435304, 7.5958561e-05], [36.2290202, 4, 6, 54.3435304, 7.5958561e-05]]
    assert_array_almost_equal(r1['Intercept']['stat'].values, a, decimal=6)
    L = ['Intercept', 'Drug[T.Trimethaphan]', 'Drug[T.placebo]']
    M = ['Histamine1', 'Histamine3', 'Histamine5']
    r1 = r.mv_test(hypotheses=[['a', L, M]])
    a = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
    assert_array_almost_equal(r1['a']['contrast_L'], a, decimal=10)
    a = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert_array_almost_equal(r1['a']['transform_M'].T, a, decimal=10)