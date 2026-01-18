import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, scale
from sklearn.utils import parallel_backend
from sklearn.utils._testing import _convert_container
@pytest.mark.parametrize('input_type', ['array', 'dataframe'])
def test_permutation_importance_large_memmaped_data(input_type):
    n_samples, n_features = (int(50000.0), 4)
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=0)
    assert X.nbytes > 1000000.0
    X = _convert_container(X, input_type)
    clf = DummyClassifier(strategy='prior').fit(X, y)
    n_repeats = 5
    r = permutation_importance(clf, X, y, n_repeats=n_repeats, n_jobs=2)
    expected_importances = np.zeros((n_features, n_repeats))
    assert_allclose(expected_importances, r.importances)