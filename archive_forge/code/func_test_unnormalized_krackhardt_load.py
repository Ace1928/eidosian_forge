import pytest
import networkx as nx
def test_unnormalized_krackhardt_load(self):
    G = self.K
    c = nx.load_centrality(G, normalized=False)
    d = {0: 1.667, 1: 1.667, 2: 0.0, 3: 7.333, 4: 0.0, 5: 16.667, 6: 16.667, 7: 28.0, 8: 16.0, 9: 0.0}
    for n in sorted(G):
        assert c[n] == pytest.approx(d[n], abs=0.001)