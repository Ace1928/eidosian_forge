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
@pytest.mark.parametrize('solver', ['auto', 'dense', 'arpack', 'randomized'])
def test_precomputed_kernel_not_psd(solver):
    """Check how KernelPCA works with non-PSD kernels depending on n_components

    Tests for all methods what happens with a non PSD gram matrix (this
    can happen in an isomap scenario, or with custom kernel functions, or
    maybe with ill-posed datasets).

    When ``n_component`` is large enough to capture a negative eigenvalue, an
    error should be raised. Otherwise, KernelPCA should run without error
    since the negative eigenvalues are not selected.
    """
    K = [[4.48, -1.0, 8.07, 2.33, 2.33, 2.33, -5.76, -12.78], [-1.0, -6.48, 4.5, -1.24, -1.24, -1.24, -0.81, 7.49], [8.07, 4.5, 15.48, 2.09, 2.09, 2.09, -11.1, -23.23], [2.33, -1.24, 2.09, 4.0, -3.65, -3.65, 1.02, -0.9], [2.33, -1.24, 2.09, -3.65, 4.0, -3.65, 1.02, -0.9], [2.33, -1.24, 2.09, -3.65, -3.65, 4.0, 1.02, -0.9], [-5.76, -0.81, -11.1, 1.02, 1.02, 1.02, 4.86, 9.75], [-12.78, 7.49, -23.23, -0.9, -0.9, -0.9, 9.75, 21.46]]
    kpca = KernelPCA(kernel='precomputed', eigen_solver=solver, n_components=7)
    with pytest.raises(ValueError, match='There are significant negative eigenvalues'):
        kpca.fit(K)
    kpca = KernelPCA(kernel='precomputed', eigen_solver=solver, n_components=2)
    if solver == 'randomized':
        with pytest.raises(ValueError, match='There are significant negative eigenvalues'):
            kpca.fit(K)
    else:
        kpca.fit(K)