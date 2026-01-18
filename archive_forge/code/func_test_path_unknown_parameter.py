import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('path_func', [enet_path, lasso_path])
def test_path_unknown_parameter(path_func):
    """Check that passing parameter not used by the coordinate descent solver
    will raise an error."""
    X, y, _, _ = build_dataset(n_samples=50, n_features=20)
    err_msg = 'Unexpected parameters in params'
    with pytest.raises(ValueError, match=err_msg):
        path_func(X, y, normalize=True, fit_intercept=True)