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
def test_rfecv_std_and_mean(global_random_seed):
    generator = check_random_state(global_random_seed)
    iris = load_iris()
    X = np.c_[iris.data, generator.normal(size=(len(iris.data), 6))]
    y = iris.target
    rfecv = RFECV(estimator=SVC(kernel='linear'))
    rfecv.fit(X, y)
    n_split_keys = len(rfecv.cv_results_) - 2
    split_keys = [f'split{i}_test_score' for i in range(n_split_keys)]
    cv_scores = np.asarray([rfecv.cv_results_[key] for key in split_keys])
    expected_mean = np.mean(cv_scores, axis=0)
    expected_std = np.std(cv_scores, axis=0)
    assert_allclose(rfecv.cv_results_['mean_test_score'], expected_mean)
    assert_allclose(rfecv.cv_results_['std_test_score'], expected_std)