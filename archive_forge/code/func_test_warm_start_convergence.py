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
def test_warm_start_convergence():
    X, y, _, _ = build_dataset()
    model = ElasticNet(alpha=0.001, tol=0.001).fit(X, y)
    n_iter_reference = model.n_iter_
    assert n_iter_reference > 2
    model.fit(X, y)
    n_iter_cold_start = model.n_iter_
    assert n_iter_cold_start == n_iter_reference
    model.set_params(warm_start=True)
    model.fit(X, y)
    n_iter_warm_start = model.n_iter_
    assert n_iter_warm_start == 1