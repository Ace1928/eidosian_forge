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
@pytest.mark.parametrize('stacker, X, y', [(StackingClassifier(estimators=[('lr', LogisticRegression()), ('svm', SVC())], cv='prefit'), X_iris, y_iris), (StackingRegressor(estimators=[('lr', LinearRegression()), ('svm', LinearSVR(dual='auto'))], cv='prefit'), X_diabetes, y_diabetes)])
def test_stacking_prefit_error(stacker, X, y):
    with pytest.raises(NotFittedError):
        stacker.fit(X, y)