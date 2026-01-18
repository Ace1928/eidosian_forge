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
def test_pairwise_n_features_in():
    """Check the n_features_in_ attributes of the meta and base estimators

    When the training data is a regular design matrix, everything is intuitive.
    However, when the training data is a precomputed kernel matrix, the
    multiclass strategy can resample the kernel matrix of the underlying base
    estimator both row-wise and column-wise and this has a non-trivial impact
    on the expected value for the n_features_in_ of both the meta and the base
    estimators.
    """
    X, y = (iris.data, iris.target)
    assert y[-1] == 0
    X = X[:-1]
    y = y[:-1]
    assert X.shape == (149, 4)
    clf_notprecomputed = svm.SVC(kernel='linear').fit(X, y)
    assert clf_notprecomputed.n_features_in_ == 4
    ovr_notprecomputed = OneVsRestClassifier(clf_notprecomputed).fit(X, y)
    assert ovr_notprecomputed.n_features_in_ == 4
    for est in ovr_notprecomputed.estimators_:
        assert est.n_features_in_ == 4
    ovo_notprecomputed = OneVsOneClassifier(clf_notprecomputed).fit(X, y)
    assert ovo_notprecomputed.n_features_in_ == 4
    assert ovo_notprecomputed.n_classes_ == 3
    assert len(ovo_notprecomputed.estimators_) == 3
    for est in ovo_notprecomputed.estimators_:
        assert est.n_features_in_ == 4
    K = X @ X.T
    assert K.shape == (149, 149)
    clf_precomputed = svm.SVC(kernel='precomputed').fit(K, y)
    assert clf_precomputed.n_features_in_ == 149
    ovr_precomputed = OneVsRestClassifier(clf_precomputed).fit(K, y)
    assert ovr_precomputed.n_features_in_ == 149
    assert ovr_precomputed.n_classes_ == 3
    assert len(ovr_precomputed.estimators_) == 3
    for est in ovr_precomputed.estimators_:
        assert est.n_features_in_ == 149
    ovo_precomputed = OneVsOneClassifier(clf_precomputed).fit(K, y)
    assert ovo_precomputed.n_features_in_ == 149
    assert ovr_precomputed.n_classes_ == 3
    assert len(ovr_precomputed.estimators_) == 3
    assert ovo_precomputed.estimators_[0].n_features_in_ == 99
    assert ovo_precomputed.estimators_[1].n_features_in_ == 99
    assert ovo_precomputed.estimators_[2].n_features_in_ == 100