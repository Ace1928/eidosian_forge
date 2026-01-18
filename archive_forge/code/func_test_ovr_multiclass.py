from re import escape
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn import datasets, svm
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.multiclass import (
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import (
from sklearn.utils._mocking import CheckingClassifier
from sklearn.utils._testing import assert_almost_equal, assert_array_equal
from sklearn.utils.fixes import (
from sklearn.utils.multiclass import check_classification_targets, type_of_target
def test_ovr_multiclass():
    X = np.array([[0, 0, 5], [0, 5, 0], [3, 0, 0], [0, 0, 6], [6, 0, 0]])
    y = ['eggs', 'spam', 'ham', 'eggs', 'ham']
    Y = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]])
    classes = set('ham eggs spam'.split())
    for base_clf in (MultinomialNB(), LinearSVC(dual='auto', random_state=0), LinearRegression(), Ridge(), ElasticNet()):
        clf = OneVsRestClassifier(base_clf).fit(X, y)
        assert set(clf.classes_) == classes
        y_pred = clf.predict(np.array([[0, 0, 4]]))[0]
        assert_array_equal(y_pred, ['eggs'])
        clf = OneVsRestClassifier(base_clf).fit(X, Y)
        y_pred = clf.predict([[0, 0, 4]])[0]
        assert_array_equal(y_pred, [0, 0, 1])