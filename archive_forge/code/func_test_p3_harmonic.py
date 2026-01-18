import pytest
import networkx as nx
from networkx.algorithms.centrality import harmonic_centrality
def test_p3_harmonic(self):
    c = harmonic_centrality(self.P3)
    d = {0: 1.5, 1: 2, 2: 1.5}
    for n in sorted(self.P3):
        assert c[n] == pytest.approx(d[n], abs=0.001)