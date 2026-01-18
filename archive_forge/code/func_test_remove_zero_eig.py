import warnings
import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_blobs, make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Perceptron
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import _check_psd_eigenvalues
def test_remove_zero_eig():
    """Check that the ``remove_zero_eig`` parameter works correctly.

    Tests that the null-space (Zero) eigenvalues are removed when
    remove_zero_eig=True, whereas they are not by default.
    """
    X = np.array([[1 - 1e-30, 1], [1, 1], [1, 1 - 1e-20]])
    kpca = KernelPCA()
    Xt = kpca.fit_transform(X)
    assert Xt.shape == (3, 0)
    kpca = KernelPCA(n_components=2)
    Xt = kpca.fit_transform(X)
    assert Xt.shape == (3, 2)
    kpca = KernelPCA(n_components=2, remove_zero_eig=True)
    Xt = kpca.fit_transform(X)
    assert Xt.shape == (3, 0)