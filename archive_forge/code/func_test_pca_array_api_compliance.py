import re
import warnings
import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_array_equal
from sklearn import config_context, datasets
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _assess_dimension, _infer_dimension
from sklearn.utils._array_api import (
from sklearn.utils._array_api import device as array_device
from sklearn.utils._testing import _array_api_for_tests, assert_allclose
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('array_namespace, device, dtype_name', yield_namespace_device_dtype_combinations())
@pytest.mark.parametrize('check', [check_array_api_input_and_values, check_array_api_get_precision], ids=_get_check_estimator_ids)
@pytest.mark.parametrize('estimator', [PCA(n_components=2, svd_solver='full'), PCA(n_components=0.1, svd_solver='full', whiten=True), PCA(n_components=2, svd_solver='randomized', power_iteration_normalizer='QR', random_state=0)], ids=_get_check_estimator_ids)
def test_pca_array_api_compliance(estimator, check, array_namespace, device, dtype_name):
    name = estimator.__class__.__name__
    check(name, estimator, array_namespace, device=device, dtype_name=dtype_name)