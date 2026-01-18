import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_perceptron_correctness():
    y_bin = y.copy()
    y_bin[y != 1] = -1
    clf1 = MyPerceptron(n_iter=2)
    clf1.fit(X, y_bin)
    clf2 = Perceptron(max_iter=2, shuffle=False, tol=None)
    clf2.fit(X, y_bin)
    assert_array_almost_equal(clf1.w, clf2.coef_.ravel())