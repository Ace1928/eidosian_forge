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
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize('stacker, X, y', [(StackingClassifier(estimators=[('lr', LogisticRegression()), ('svm', LinearSVC(dual='auto', random_state=42))], final_estimator=LogisticRegression()), *load_breast_cancer(return_X_y=True)), (StackingRegressor(estimators=[('lr', LinearRegression()), ('svm', LinearSVR(dual='auto', random_state=42))], final_estimator=LinearRegression()), X_diabetes, y_diabetes)], ids=['StackingClassifier', 'StackingRegressor'])
def test_stacking_cv_influence(stacker, X, y):
    stacker_cv_3 = clone(stacker)
    stacker_cv_5 = clone(stacker)
    stacker_cv_3.set_params(cv=3)
    stacker_cv_5.set_params(cv=5)
    stacker_cv_3.fit(X, y)
    stacker_cv_5.fit(X, y)
    for est_cv_3, est_cv_5 in zip(stacker_cv_3.estimators_, stacker_cv_5.estimators_):
        assert_allclose(est_cv_3.coef_, est_cv_5.coef_)
    with pytest.raises(AssertionError, match='Not equal'):
        assert_allclose(stacker_cv_3.final_estimator_.coef_, stacker_cv_5.final_estimator_.coef_)