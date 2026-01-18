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
def test_permutation_importance_no_weights_scoring_function():

    def my_scorer(estimator, X, y):
        return 1
    x = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    w = np.array([1, 1])
    lr = LinearRegression()
    lr.fit(x, y)
    try:
        permutation_importance(lr, x, y, random_state=1, scoring=my_scorer, n_repeats=1)
    except TypeError:
        pytest.fail('permutation_test raised an error when using a scorer function that does not accept sample_weight even though sample_weight was None')
    with pytest.raises(TypeError):
        permutation_importance(lr, x, y, random_state=1, scoring=my_scorer, n_repeats=1, sample_weight=w)