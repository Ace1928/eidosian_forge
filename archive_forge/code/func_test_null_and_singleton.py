from math import sqrt
import pytest
import networkx as nx
@pytest.mark.parametrize('method', methods)
def test_null_and_singleton(self, method):
    G = nx.Graph()
    pytest.raises(nx.NetworkXError, nx.algebraic_connectivity, G, method=method)
    pytest.raises(nx.NetworkXError, nx.fiedler_vector, G, method=method)
    G.add_edge(0, 0)
    pytest.raises(nx.NetworkXError, nx.algebraic_connectivity, G, method=method)
    pytest.raises(nx.NetworkXError, nx.fiedler_vector, G, method=method)