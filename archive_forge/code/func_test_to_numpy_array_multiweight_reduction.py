import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
@pytest.mark.parametrize('func, expected', ((min, -1), (max, 10), (sum, 11), (np.mean, 11 / 3), (np.median, 2)))
def test_to_numpy_array_multiweight_reduction(func, expected):
    """Test various functions for reducing multiedge weights."""
    G = nx.MultiDiGraph()
    weights = [-1, 2, 10.0]
    for w in weights:
        G.add_edge(0, 1, weight=w)
    A = nx.to_numpy_array(G, multigraph_weight=func, dtype=float)
    assert np.allclose(A, [[0, expected], [0, 0]])
    A = nx.to_numpy_array(G.to_undirected(), multigraph_weight=func, dtype=float)
    assert np.allclose(A, [[0, expected], [expected, 0]])