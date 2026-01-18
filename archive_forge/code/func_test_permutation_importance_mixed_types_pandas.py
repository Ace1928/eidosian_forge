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
def test_permutation_importance_mixed_types_pandas():
    pd = pytest.importorskip('pandas')
    rng = np.random.RandomState(42)
    n_repeats = 5
    X = pd.DataFrame({'col1': [1.0, 2.0, 3.0, np.nan], 'col2': ['a', 'b', 'a', 'b']})
    y = np.array([0, 1, 0, 1])
    num_preprocess = make_pipeline(SimpleImputer(), StandardScaler())
    preprocess = ColumnTransformer([('num', num_preprocess, ['col1']), ('cat', OneHotEncoder(), ['col2'])])
    clf = make_pipeline(preprocess, LogisticRegression(solver='lbfgs'))
    clf.fit(X, y)
    result = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=rng)
    assert result.importances.shape == (X.shape[1], n_repeats)
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])