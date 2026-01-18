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
def test_warm_start_convergence_with_regularizer_decrement():
    X, y = load_diabetes(return_X_y=True)
    final_alpha = 1e-05
    low_reg_model = ElasticNet(alpha=final_alpha).fit(X, y)
    high_reg_model = ElasticNet(alpha=final_alpha * 10).fit(X, y)
    assert low_reg_model.n_iter_ > high_reg_model.n_iter_
    warm_low_reg_model = deepcopy(high_reg_model)
    warm_low_reg_model.set_params(warm_start=True, alpha=final_alpha)
    warm_low_reg_model.fit(X, y)
    assert low_reg_model.n_iter_ > warm_low_reg_model.n_iter_