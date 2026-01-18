from math import ceil
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris, make_blobs
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
@pytest.mark.parametrize('base_estimator', [KNeighborsClassifier(), SVC(gamma='scale', probability=True, random_state=0)])
@pytest.mark.parametrize('y', [y_train_missing_labels, y_train_missing_strings])
def test_zero_iterations(base_estimator, y):
    clf1 = SelfTrainingClassifier(base_estimator, max_iter=0)
    clf1.fit(X_train, y)
    clf2 = base_estimator.fit(X_train[:n_labeled_samples], y[:n_labeled_samples])
    assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))
    assert clf1.termination_condition_ == 'max_iter'