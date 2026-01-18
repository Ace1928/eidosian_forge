import pytest
import networkx as nx
def test_not_strongly_connected(self):
    b = nx.load_centrality(self.D)
    result = {0: 5.0 / 12, 1: 1.0 / 4, 2: 1.0 / 12, 3: 1.0 / 4, 4: 0.0}
    for n in sorted(self.D):
        assert result[n] == pytest.approx(b[n], abs=0.001)
        assert result[n] == pytest.approx(nx.load_centrality(self.D, n), abs=0.001)