import pytest
import networkx as nx
def test_peng_square_clustering(self):
    """Test eq2 for figure 1 Peng et al (2008)"""
    G = nx.Graph([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (3, 6)])
    assert nx.square_clustering(G, [1])[1] == 1 / 3