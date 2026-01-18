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
@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + CSC_CONTAINERS + COO_CONTAINERS + DOK_CONTAINERS + LIL_CONTAINERS)
def test_ovr_fit_predict_sparse(sparse_container):
    base_clf = MultinomialNB(alpha=1)
    X, Y = datasets.make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, n_labels=3, length=50, allow_unlabeled=True, random_state=0)
    X_train, Y_train = (X[:80], Y[:80])
    X_test = X[80:]
    clf = OneVsRestClassifier(base_clf).fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    clf_sprs = OneVsRestClassifier(base_clf).fit(X_train, sparse_container(Y_train))
    Y_pred_sprs = clf_sprs.predict(X_test)
    assert clf.multilabel_
    assert sp.issparse(Y_pred_sprs)
    assert_array_equal(Y_pred_sprs.toarray(), Y_pred)
    Y_proba = clf_sprs.predict_proba(X_test)
    pred = Y_proba > 0.5
    assert_array_equal(pred, Y_pred_sprs.toarray())
    clf = svm.SVC()
    clf_sprs = OneVsRestClassifier(clf).fit(X_train, sparse_container(Y_train))
    dec_pred = (clf_sprs.decision_function(X_test) > 0).astype(int)
    assert_array_equal(dec_pred, clf_sprs.predict(X_test).toarray())