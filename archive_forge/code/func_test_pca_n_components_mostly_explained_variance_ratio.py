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
def test_pca_n_components_mostly_explained_variance_ratio():
    X, y = load_iris(return_X_y=True)
    pca1 = PCA().fit(X, y)
    n_components = pca1.explained_variance_ratio_.cumsum()[-2]
    pca2 = PCA(n_components=n_components).fit(X, y)
    assert pca2.n_components_ == X.shape[1]