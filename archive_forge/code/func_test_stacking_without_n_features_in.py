from unittest.mock import Mock
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import (
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC, LinearSVC, LinearSVR
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('make_dataset, Stacking, Estimator', [(make_classification, StackingClassifier, LogisticRegression), (make_regression, StackingRegressor, LinearRegression)])
def test_stacking_without_n_features_in(make_dataset, Stacking, Estimator):

    class MyEstimator(Estimator):
        """Estimator without n_features_in_"""

        def fit(self, X, y):
            super().fit(X, y)
            del self.n_features_in_
    X, y = make_dataset(random_state=0, n_samples=100)
    stacker = Stacking(estimators=[('lr', MyEstimator())])
    msg = f'{Stacking.__name__} object has no attribute n_features_in_'
    with pytest.raises(AttributeError, match=msg):
        stacker.n_features_in_
    stacker.fit(X, y)
    msg = "'MyEstimator' object has no attribute 'n_features_in_'"
    with pytest.raises(AttributeError, match=msg):
        stacker.n_features_in_