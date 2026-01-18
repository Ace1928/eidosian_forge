import pytest
import networkx as nx
from networkx.algorithms.centrality import harmonic_centrality
def test_p4_harmonic_subset(self):
    c = harmonic_centrality(self.P4, nbunch=[2, 3], sources=[0, 1])
    d = {2: 1.5, 3: 0.8333333}
    for n in [2, 3]:
        assert c[n] == pytest.approx(d[n], abs=0.001)