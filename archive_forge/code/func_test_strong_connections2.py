import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from scipy.sparse import csgraph, csr_array
def test_strong_connections2():
    X = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
    n_components, labels = csgraph.connected_components(X, directed=True, connection='strong')
    assert_equal(n_components, 5)
    labels.sort()
    assert_array_almost_equal(labels, [0, 1, 2, 2, 3, 4])