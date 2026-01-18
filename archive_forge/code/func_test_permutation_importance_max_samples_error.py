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
def test_permutation_importance_max_samples_error():
    """Check that a proper error message is raised when `max_samples` is not
    set to a valid input value.
    """
    X = np.array([(1.0, 2.0, 3.0, 4.0)]).T
    y = np.array([0, 1, 0, 1])
    clf = LogisticRegression()
    clf.fit(X, y)
    err_msg = 'max_samples must be <= n_samples'
    with pytest.raises(ValueError, match=err_msg):
        permutation_importance(clf, X, y, max_samples=5)