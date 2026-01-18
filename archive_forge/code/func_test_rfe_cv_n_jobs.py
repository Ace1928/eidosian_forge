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
def test_rfe_cv_n_jobs(global_random_seed):
    generator = check_random_state(global_random_seed)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target
    rfecv = RFECV(estimator=SVC(kernel='linear'))
    rfecv.fit(X, y)
    rfecv_ranking = rfecv.ranking_
    rfecv_cv_results_ = rfecv.cv_results_
    rfecv.set_params(n_jobs=2)
    rfecv.fit(X, y)
    assert_array_almost_equal(rfecv.ranking_, rfecv_ranking)
    assert rfecv_cv_results_.keys() == rfecv.cv_results_.keys()
    for key in rfecv_cv_results_.keys():
        assert rfecv_cv_results_[key] == pytest.approx(rfecv.cv_results_[key])