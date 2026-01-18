import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('n_features_to_select', (2, 3))
def test_unsupervised_model_fit(n_features_to_select):
    X, y = make_blobs(n_features=4)
    sfs = SequentialFeatureSelector(KMeans(n_init=1), n_features_to_select=n_features_to_select)
    sfs.fit(X)
    assert sfs.transform(X).shape[1] == n_features_to_select