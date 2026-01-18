import numpy as np
import pytest
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier, PassiveAggressiveRegressor
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_wrong_class_weight_label():
    X2 = np.array([[-1.0, -1.0], [-1.0, 0], [-0.8, -1.0], [1.0, 1.0], [1.0, 0.0]])
    y2 = [1, 1, 1, -1, -1]
    clf = PassiveAggressiveClassifier(class_weight={0: 0.5}, max_iter=100)
    with pytest.raises(ValueError):
        clf.fit(X2, y2)