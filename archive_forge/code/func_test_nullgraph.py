from math import sqrt
import pytest
import networkx as nx
@pytest.mark.parametrize('graph', _graphs)
def test_nullgraph(self, graph):
    G = graph()
    pytest.raises(nx.NetworkXError, nx.spectral_ordering, G)