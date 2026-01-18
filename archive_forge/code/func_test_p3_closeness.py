import pytest
import networkx as nx
def test_p3_closeness(self):
    c = nx.closeness_centrality(self.P3)
    d = {0: 0.667, 1: 1.0, 2: 0.667}
    for n in sorted(self.P3):
        assert c[n] == pytest.approx(d[n], abs=0.001)