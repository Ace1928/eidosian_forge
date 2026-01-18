import warnings
from itertools import product
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets
from sklearn.datasets import (
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._ridge import (
from sklearn.metrics import get_scorer, make_scorer, mean_squared_error
from sklearn.model_selection import (
from sklearn.preprocessing import minmax_scale
from sklearn.utils import _IS_32BIT, check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('Classifier, params', [(RidgeClassifier, {}), (RidgeClassifierCV, {'cv': None}), (RidgeClassifierCV, {'cv': 3})])
def test_ridgeclassifier_multilabel(Classifier, params):
    """Check that multilabel classification is supported and give meaningful
    results."""
    X, y = make_multilabel_classification(n_classes=1, random_state=0)
    y = y.reshape(-1, 1)
    Y = np.concatenate([y, y], axis=1)
    clf = Classifier(**params).fit(X, Y)
    Y_pred = clf.predict(X)
    assert Y_pred.shape == Y.shape
    assert_array_equal(Y_pred[:, 0], Y_pred[:, 1])
    Ridge(solver='sag').fit(X, y)