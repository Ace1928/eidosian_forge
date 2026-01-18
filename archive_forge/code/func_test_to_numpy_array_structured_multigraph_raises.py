import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
@pytest.mark.parametrize('graph_type', (nx.MultiGraph, nx.MultiDiGraph))
def test_to_numpy_array_structured_multigraph_raises(graph_type):
    G = nx.path_graph(3, create_using=graph_type)
    dtype = np.dtype([('weight', int), ('cost', int)])
    with pytest.raises(nx.NetworkXError, match='Structured arrays are not supported'):
        nx.to_numpy_array(G, dtype=dtype, weight=None)