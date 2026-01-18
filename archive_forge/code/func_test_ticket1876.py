import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from scipy.sparse import csgraph, csr_array
def test_ticket1876():
    g = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0]])
    n_components, labels = csgraph.connected_components(g, connection='strong')
    assert_equal(n_components, 2)
    assert_equal(labels[0], labels[1])
    assert_equal(labels[2], labels[3])