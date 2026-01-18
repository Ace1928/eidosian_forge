import warnings
import numpy as np
import pytest
from sklearn.cluster import AffinityPropagation, affinity_propagation
from sklearn.cluster._affinity_propagation import _equal_similarities_and_preferences
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_sparse_input_for_fit_predict(csr_container):
    af = AffinityPropagation(affinity='euclidean', random_state=42)
    rng = np.random.RandomState(42)
    X = csr_container(rng.randint(0, 2, size=(5, 5)))
    labels = af.fit_predict(X)
    assert_array_equal(labels, (0, 1, 1, 2, 3))