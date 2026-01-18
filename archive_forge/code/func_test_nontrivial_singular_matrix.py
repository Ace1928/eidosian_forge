import scipy.stats
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_almost_equal
from patsy import dmatrices  # pylint: disable=E0611
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from .results.results_quantile_regression import (
def test_nontrivial_singular_matrix():
    x_one = np.random.random(1000)
    x_two = np.random.random(1000) * 10
    x_three = np.random.random(1000)
    intercept = np.ones(1000)
    y = np.random.random(1000) * 5
    X = np.column_stack((intercept, x_one, x_two, x_three, x_one))
    assert np.linalg.matrix_rank(X) < X.shape[1]
    res_singular = QuantReg(y, X).fit(0.5)
    assert len(res_singular.params) == X.shape[1]
    assert np.linalg.matrix_rank(res_singular.cov_params()) == X.shape[1] - 1
    res_ns = QuantReg(y, X[:, :-1]).fit(0.5)
    assert_allclose(res_singular.fittedvalues, res_ns.fittedvalues, rtol=0.01)