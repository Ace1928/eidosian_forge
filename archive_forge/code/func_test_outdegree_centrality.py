import pytest
import networkx as nx
def test_outdegree_centrality(self):
    d = nx.out_degree_centrality(self.G)
    exact = {0: 0.125, 1: 0.125, 2: 0.125, 3: 0.125, 4: 0.125, 5: 0.375, 6: 0.0, 7: 0.0, 8: 0.0}
    for n, dc in d.items():
        assert exact[n] == pytest.approx(dc, abs=1e-07)