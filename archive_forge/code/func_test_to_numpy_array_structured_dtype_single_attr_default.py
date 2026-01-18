import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
def test_to_numpy_array_structured_dtype_single_attr_default():
    G = nx.path_graph(3)
    dtype = np.dtype([('weight', float)])
    A = nx.to_numpy_array(G, dtype=dtype, weight=None)
    expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    npt.assert_array_equal(A['weight'], expected)