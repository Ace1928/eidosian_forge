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
def test_ovo_ties2():
    X = np.array([[1, 2], [2, 1], [-2, 1], [-2, -1]])
    y_ref = np.array([2, 0, 1, 2])
    for i in range(3):
        y = (y_ref + i) % 3
        multi_clf = OneVsOneClassifier(Perceptron(shuffle=False, max_iter=4, tol=None))
        ovo_prediction = multi_clf.fit(X, y).predict(X)
        assert ovo_prediction[0] == i % 3