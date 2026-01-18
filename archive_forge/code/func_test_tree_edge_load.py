import pytest
import networkx as nx
def test_tree_edge_load(self):
    G = self.T
    c = nx.edge_load_centrality(G)
    d = {(0, 1): 24.0, (0, 2): 24.0, (1, 3): 12.0, (1, 4): 12.0, (2, 5): 12.0, (2, 6): 12.0}
    for n in G.edges():
        assert c[n] == pytest.approx(d[n], abs=0.001)