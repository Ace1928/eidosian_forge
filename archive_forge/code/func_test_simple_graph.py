import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.csgraph._flow import (
@pytest.mark.parametrize('method', methods)
def test_simple_graph(method):
    graph = csr_matrix([[0, 5], [0, 0]])
    res = maximum_flow(graph, 0, 1, method=method)
    assert res.flow_value == 5
    expected_flow = np.array([[0, 5], [-5, 0]])
    assert_array_equal(res.flow.toarray(), expected_flow)