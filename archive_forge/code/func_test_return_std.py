from math import log
import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import ARDRegression, BayesianRidge, Ridge
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.extmath import fast_logdet
@pytest.mark.parametrize('constructor_name', ['array', 'dataframe'])
def test_return_std(constructor_name):

    def f(X):
        return np.dot(X, w) + b

    def f_noise(X, noise_mult):
        return f(X) + np.random.randn(X.shape[0]) * noise_mult
    d = 5
    n_train = 50
    n_test = 10
    w = np.array([1.0, 0.0, 1.0, -1.0, 0.0])
    b = 1.0
    X = np.random.random((n_train, d))
    X = _convert_container(X, constructor_name)
    X_test = np.random.random((n_test, d))
    X_test = _convert_container(X_test, constructor_name)
    for decimal, noise_mult in enumerate([1, 0.1, 0.01]):
        y = f_noise(X, noise_mult)
        m1 = BayesianRidge()
        m1.fit(X, y)
        y_mean1, y_std1 = m1.predict(X_test, return_std=True)
        assert_array_almost_equal(y_std1, noise_mult, decimal=decimal)
        m2 = ARDRegression()
        m2.fit(X, y)
        y_mean2, y_std2 = m2.predict(X_test, return_std=True)
        assert_array_almost_equal(y_std2, noise_mult, decimal=decimal)