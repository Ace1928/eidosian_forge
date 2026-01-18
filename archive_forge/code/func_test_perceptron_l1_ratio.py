import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_perceptron_l1_ratio():
    """Check that `l1_ratio` has an impact when `penalty='elasticnet'`"""
    clf1 = Perceptron(l1_ratio=0, penalty='elasticnet')
    clf1.fit(X, y)
    clf2 = Perceptron(l1_ratio=0.15, penalty='elasticnet')
    clf2.fit(X, y)
    assert clf1.score(X, y) != clf2.score(X, y)
    clf_l1 = Perceptron(penalty='l1').fit(X, y)
    clf_elasticnet = Perceptron(l1_ratio=1, penalty='elasticnet').fit(X, y)
    assert_allclose(clf_l1.coef_, clf_elasticnet.coef_)
    clf_l2 = Perceptron(penalty='l2').fit(X, y)
    clf_elasticnet = Perceptron(l1_ratio=0, penalty='elasticnet').fit(X, y)
    assert_allclose(clf_l2.coef_, clf_elasticnet.coef_)