from io import StringIO
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises
from scipy.sparse.csgraph import (shortest_path, dijkstra, johnson,
import scipy.sparse
from scipy.io import mmread
import pytest
def test_shortest_path_indices():
    indices = np.arange(4)

    def check(func, indshape):
        outshape = indshape + (5,)
        SP = func(directed_G, directed=False, indices=indices.reshape(indshape))
        assert_array_almost_equal(SP, undirected_SP[indices].reshape(outshape))
    for indshape in [(4,), (4, 1), (2, 2)]:
        for func in (dijkstra, bellman_ford, johnson, shortest_path):
            check(func, indshape)
    assert_raises(ValueError, shortest_path, directed_G, method='FW', indices=indices)