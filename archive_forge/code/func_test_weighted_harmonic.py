import pytest
import networkx as nx
from networkx.algorithms.centrality import harmonic_centrality
def test_weighted_harmonic(self):
    XG = nx.DiGraph()
    XG.add_weighted_edges_from([('a', 'b', 10), ('d', 'c', 5), ('a', 'c', 1), ('e', 'f', 2), ('f', 'c', 1), ('a', 'f', 3)])
    c = harmonic_centrality(XG, distance='weight')
    d = {'a': 0, 'b': 0.1, 'c': 2.533, 'd': 0, 'e': 0, 'f': 0.83333}
    for n in sorted(XG):
        assert c[n] == pytest.approx(d[n], abs=0.001)