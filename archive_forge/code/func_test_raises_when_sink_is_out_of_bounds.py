import numpy as np
from numpy.testing import assert_array_equal
import pytest
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.csgraph._flow import (
@pytest.mark.parametrize('method', methods)
@pytest.mark.parametrize('sink', [-1, 2, 3])
def test_raises_when_sink_is_out_of_bounds(sink, method):
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, 0, sink, method=method)