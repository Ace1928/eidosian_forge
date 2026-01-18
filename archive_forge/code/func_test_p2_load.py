import pytest
import networkx as nx
def test_p2_load(self):
    G = nx.path_graph(2)
    c = nx.load_centrality(G)
    d = {0: 0.0, 1: 0.0}
    for n in sorted(G):
        assert c[n] == pytest.approx(d[n], abs=0.001)