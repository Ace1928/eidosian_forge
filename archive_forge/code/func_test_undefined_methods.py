import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal
from sklearn.utils.fixes import CSR_CONTAINERS
def test_undefined_methods():
    clf = Perceptron(max_iter=100)
    for meth in ('predict_proba', 'predict_log_proba'):
        with pytest.raises(AttributeError):
            getattr(clf, meth)