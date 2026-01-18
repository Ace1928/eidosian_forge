import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('container', CSR_CONTAINERS + [np.array])
def test_perceptron_accuracy(container):
    data = container(X)
    clf = Perceptron(max_iter=100, tol=None, shuffle=False)
    clf.fit(data, y)
    score = clf.score(data, y)
    assert score > 0.7