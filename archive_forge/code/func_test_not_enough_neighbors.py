import pytest
from networkx import NetworkXError, cycle_graph
from networkx.algorithms.bipartite import complete_bipartite_graph, node_redundancy
def test_not_enough_neighbors():
    with pytest.raises(NetworkXError):
        G = complete_bipartite_graph(1, 2)
        node_redundancy(G)