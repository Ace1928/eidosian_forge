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
def test_rfe_features_importance():
    generator = check_random_state(0)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target
    clf = RandomForestClassifier(n_estimators=20, random_state=generator, max_depth=2)
    rfe = RFE(estimator=clf, n_features_to_select=4, step=0.1)
    rfe.fit(X, y)
    assert len(rfe.ranking_) == X.shape[1]
    clf_svc = SVC(kernel='linear')
    rfe_svc = RFE(estimator=clf_svc, n_features_to_select=4, step=0.1)
    rfe_svc.fit(X, y)
    assert_array_equal(rfe.get_support(), rfe_svc.get_support())