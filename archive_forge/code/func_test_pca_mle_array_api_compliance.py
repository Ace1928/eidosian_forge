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
@pytest.mark.parametrize('check', [check_array_api_get_precision], ids=_get_check_estimator_ids)
@pytest.mark.parametrize('estimator', [PCA(n_components='mle', svd_solver='full')], ids=_get_check_estimator_ids)
def test_pca_mle_array_api_compliance(estimator, check, array_namespace, device, dtype_name):
    name = estimator.__class__.__name__
    check(name, estimator, array_namespace, device=device, dtype_name=dtype_name)
    xp = _array_api_for_tests(array_namespace, device)
    X, y = make_classification(random_state=42)
    X = X.astype(dtype_name, copy=False)
    atol = _atol_for_type(X.dtype)
    est = clone(estimator)
    X_xp = xp.asarray(X, device=device)
    y_xp = xp.asarray(y, device=device)
    est.fit(X, y)
    components_np = est.components_
    explained_variance_np = est.explained_variance_
    est_xp = clone(est)
    with config_context(array_api_dispatch=True):
        est_xp.fit(X_xp, y_xp)
        components_xp = est_xp.components_
        assert array_device(components_xp) == array_device(X_xp)
        components_xp_np = _convert_to_numpy(components_xp, xp=xp)
        explained_variance_xp = est_xp.explained_variance_
        assert array_device(explained_variance_xp) == array_device(X_xp)
        explained_variance_xp_np = _convert_to_numpy(explained_variance_xp, xp=xp)
    assert components_xp_np.dtype == components_np.dtype
    assert components_xp_np.shape[1] == components_np.shape[1]
    assert explained_variance_xp_np.dtype == explained_variance_np.dtype
    min_components = min(components_xp_np.shape[0], components_np.shape[0])
    assert_allclose(explained_variance_xp_np[:min_components], explained_variance_np[:min_components], atol=atol)
    if components_xp_np.shape[0] != components_np.shape[0]:
        reference_variance = explained_variance_np[-1]
        extra_variance_np = explained_variance_np[min_components:]
        extra_variance_xp_np = explained_variance_xp_np[min_components:]
        assert all(np.abs(extra_variance_np - reference_variance) < atol)
        assert all(np.abs(extra_variance_xp_np - reference_variance) < atol)