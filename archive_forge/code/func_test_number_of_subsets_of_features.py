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
def test_number_of_subsets_of_features(global_random_seed):

    def formula1(n_features, n_features_to_select, step):
        return 1 + (n_features + step - n_features_to_select - 1) // step

    def formula2(n_features, n_features_to_select, step):
        return 1 + np.ceil((n_features - n_features_to_select) / float(step))
    n_features_list = [11, 11]
    n_features_to_select_list = [3, 3]
    step_list = [2, 3]
    for n_features, n_features_to_select, step in zip(n_features_list, n_features_to_select_list, step_list):
        generator = check_random_state(global_random_seed)
        X = generator.normal(size=(100, n_features))
        y = generator.rand(100).round()
        rfe = RFE(estimator=SVC(kernel='linear'), n_features_to_select=n_features_to_select, step=step)
        rfe.fit(X, y)
        assert np.max(rfe.ranking_) == formula1(n_features, n_features_to_select, step)
        assert np.max(rfe.ranking_) == formula2(n_features, n_features_to_select, step)
    n_features_to_select = 1
    n_features_list = [11, 10]
    step_list = [2, 2]
    for n_features, step in zip(n_features_list, step_list):
        generator = check_random_state(global_random_seed)
        X = generator.normal(size=(100, n_features))
        y = generator.rand(100).round()
        rfecv = RFECV(estimator=SVC(kernel='linear'), step=step)
        rfecv.fit(X, y)
        for key in rfecv.cv_results_.keys():
            assert len(rfecv.cv_results_[key]) == formula1(n_features, n_features_to_select, step)
            assert len(rfecv.cv_results_[key]) == formula2(n_features, n_features_to_select, step)