import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_less
@pytest.mark.parametrize('solver', SVD_SOLVERS)
def test_singular_values_consistency(solver):
    rng = np.random.RandomState(0)
    n_samples, n_features = (100, 80)
    X = rng.randn(n_samples, n_features)
    pca = TruncatedSVD(n_components=2, algorithm=solver, random_state=rng).fit(X)
    X_pca = pca.transform(X)
    assert_allclose(np.sum(pca.singular_values_ ** 2.0), np.linalg.norm(X_pca, 'fro') ** 2.0, rtol=0.01)
    assert_allclose(pca.singular_values_, np.sqrt(np.sum(X_pca ** 2.0, axis=0)), rtol=0.01)