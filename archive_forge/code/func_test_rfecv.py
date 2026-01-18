from operator import attrgetter
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.datasets import load_iris, make_friedman1
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer, make_scorer, zero_one_loss
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.utils import check_random_state
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_rfecv(csr_container):
    generator = check_random_state(0)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = list(iris.target)
    rfecv = RFECV(estimator=SVC(kernel='linear'), step=1)
    rfecv.fit(X, y)
    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == X.shape[1]
    assert len(rfecv.ranking_) == X.shape[1]
    X_r = rfecv.transform(X)
    assert_array_equal(X_r, iris.data)
    rfecv_sparse = RFECV(estimator=SVC(kernel='linear'), step=1)
    X_sparse = csr_container(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    assert_array_equal(X_r_sparse.toarray(), iris.data)
    scoring = make_scorer(zero_one_loss, greater_is_better=False)
    rfecv = RFECV(estimator=SVC(kernel='linear'), step=1, scoring=scoring)
    ignore_warnings(rfecv.fit)(X, y)
    X_r = rfecv.transform(X)
    assert_array_equal(X_r, iris.data)
    scorer = get_scorer('accuracy')
    rfecv = RFECV(estimator=SVC(kernel='linear'), step=1, scoring=scorer)
    rfecv.fit(X, y)
    X_r = rfecv.transform(X)
    assert_array_equal(X_r, iris.data)

    def test_scorer(estimator, X, y):
        return 1.0
    rfecv = RFECV(estimator=SVC(kernel='linear'), step=1, scoring=test_scorer)
    rfecv.fit(X, y)
    assert rfecv.n_features_ == 1
    rfecv = RFECV(estimator=SVC(kernel='linear'), step=2)
    rfecv.fit(X, y)
    for key in rfecv.cv_results_.keys():
        assert len(rfecv.cv_results_[key]) == 6
    assert len(rfecv.ranking_) == X.shape[1]
    X_r = rfecv.transform(X)
    assert_array_equal(X_r, iris.data)
    rfecv_sparse = RFECV(estimator=SVC(kernel='linear'), step=2)
    X_sparse = csr_container(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    assert_array_equal(X_r_sparse.toarray(), iris.data)
    rfecv_sparse = RFECV(estimator=SVC(kernel='linear'), step=0.2)
    X_sparse = csr_container(X)
    rfecv_sparse.fit(X_sparse, y)
    X_r_sparse = rfecv_sparse.transform(X_sparse)
    assert_array_equal(X_r_sparse.toarray(), iris.data)