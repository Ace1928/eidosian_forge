import pytest
import networkx as nx
from networkx.algorithms.approximation import diameter
def test_undirected_non_connected(self):
    """Test an undirected disconnected graph."""
    graph = nx.path_graph(10)
    graph.remove_edge(3, 4)
    with pytest.raises(nx.NetworkXError, match='Graph not connected.'):
        diameter(graph)