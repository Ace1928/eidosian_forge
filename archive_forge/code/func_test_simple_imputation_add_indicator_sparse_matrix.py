import io
import re
import warnings
from itertools import product
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import kstest
from sklearn import tree
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, MissingIndicator, SimpleImputer
from sklearn.impute._base import _most_frequent
from sklearn.linear_model import ARDRegression, BayesianRidge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_union
from sklearn.random_projection import _sparse_random_matrix
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
@pytest.mark.parametrize('arr_type', CSC_CONTAINERS + CSR_CONTAINERS + COO_CONTAINERS + LIL_CONTAINERS + BSR_CONTAINERS)
def test_simple_imputation_add_indicator_sparse_matrix(arr_type):
    X_sparse = arr_type([[np.nan, 1, 5], [2, np.nan, 1], [6, 3, np.nan], [1, 2, 9]])
    X_true = np.array([[3.0, 1.0, 5.0, 1.0, 0.0, 0.0], [2.0, 2.0, 1.0, 0.0, 1.0, 0.0], [6.0, 3.0, 5.0, 0.0, 0.0, 1.0], [1.0, 2.0, 9.0, 0.0, 0.0, 0.0]])
    imputer = SimpleImputer(missing_values=np.nan, add_indicator=True)
    X_trans = imputer.fit_transform(X_sparse)
    assert sparse.issparse(X_trans)
    assert X_trans.shape == X_true.shape
    assert_allclose(X_trans.toarray(), X_true)