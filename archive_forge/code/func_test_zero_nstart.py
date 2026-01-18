import math
import pytest
import networkx as nx
def test_zero_nstart(self):
    G = nx.Graph([(1, 2), (1, 3), (2, 3)])
    with pytest.raises(nx.NetworkXException, match='initial vector cannot have all zero values'):
        nx.eigenvector_centrality(G, nstart={v: 0 for v in G})