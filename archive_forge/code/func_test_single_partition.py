import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.base.distributed_estimation import _calc_grad, \
def test_single_partition():
    np.random.seed(435265)
    N = 200
    p = 10
    m = 1
    beta = np.random.normal(size=p)
    beta = beta * np.random.randint(0, 2, p)
    X = np.random.normal(size=(N, p))
    y = X.dot(beta) + np.random.normal(size=N)
    db_mod = DistributedModel(m)
    fitOLSdb = db_mod.fit(_data_gen(y, X, m), fit_kwds={'alpha': 0})
    nv_mod = DistributedModel(m, estimation_method=_est_regularized_naive, join_method=_join_naive)
    fitOLSnv = nv_mod.fit(_data_gen(y, X, m), fit_kwds={'alpha': 0})
    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit(alpha=0)
    assert_allclose(fitOLSdb.params, fitOLS.params)
    assert_allclose(fitOLSnv.params, fitOLS.params)
    nv_mod = DistributedModel(m, estimation_method=_est_regularized_naive, join_method=_join_naive)
    fitOLSnv = nv_mod.fit(_data_gen(y, X, m), fit_kwds={'alpha': 0.1})
    ols_mod = OLS(y, X)
    fitOLS = ols_mod.fit_regularized(alpha=0.1)
    assert_allclose(fitOLSnv.params, fitOLS.params)