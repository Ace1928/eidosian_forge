import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
@pytest.mark.parametrize('graph_type', (nx.Graph, nx.DiGraph))
@pytest.mark.parametrize('edge', [(0, 1), (0, 1, {'weight': 10}), (0, 1, {'weight': 5, 'flow': -4}), (0, 1, {'weight': 2.0, 'cost': 10, 'flow': -45})])
def test_to_numpy_array_structured_dtype_multiple_fields(graph_type, edge):
    G = graph_type([edge])
    dtype = np.dtype([('weight', float), ('cost', float), ('flow', float)])
    A = nx.to_numpy_array(G, dtype=dtype, weight=None)
    for attr in dtype.names:
        expected = nx.to_numpy_array(G, dtype=float, weight=attr)
        npt.assert_array_equal(A[attr], expected)