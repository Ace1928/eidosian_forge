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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_input_convergence_warning(csr_container):
    X, y, _, _ = build_dataset(n_samples=1000, n_features=500)
    with pytest.warns(ConvergenceWarning):
        ElasticNet(max_iter=1, tol=0).fit(csr_container(X, dtype=np.float32), y)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        Lasso().fit(csr_container(X, dtype=np.float32), y)