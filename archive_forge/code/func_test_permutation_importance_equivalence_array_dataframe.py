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
@pytest.mark.parametrize('n_jobs', [None, 1, 2])
@pytest.mark.parametrize('max_samples', [0.5, 1.0])
def test_permutation_importance_equivalence_array_dataframe(n_jobs, max_samples):
    pd = pytest.importorskip('pandas')
    X, y = make_regression(n_samples=100, n_features=5, random_state=0)
    X_df = pd.DataFrame(X)
    binner = KBinsDiscretizer(n_bins=3, encode='ordinal')
    cat_column = binner.fit_transform(y.reshape(-1, 1))
    X = np.hstack([X, cat_column])
    assert X.dtype.kind == 'f'
    if hasattr(pd, 'Categorical'):
        cat_column = pd.Categorical(cat_column.ravel())
    else:
        cat_column = cat_column.ravel()
    new_col_idx = len(X_df.columns)
    X_df[new_col_idx] = cat_column
    assert X_df[new_col_idx].dtype == cat_column.dtype
    X_df.index = np.arange(len(X_df)).astype(str)
    rf = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=0)
    rf.fit(X, y)
    n_repeats = 3
    importance_array = permutation_importance(rf, X, y, n_repeats=n_repeats, random_state=0, n_jobs=n_jobs, max_samples=max_samples)
    imp_min = importance_array['importances'].min()
    imp_max = importance_array['importances'].max()
    assert imp_max - imp_min > 0.3
    importance_dataframe = permutation_importance(rf, X_df, y, n_repeats=n_repeats, random_state=0, n_jobs=n_jobs, max_samples=max_samples)
    assert_allclose(importance_array['importances'], importance_dataframe['importances'])