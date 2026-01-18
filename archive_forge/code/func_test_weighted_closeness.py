import pytest
import networkx as nx
def test_weighted_closeness(self):
    edges = [('s', 'u', 10), ('s', 'x', 5), ('u', 'v', 1), ('u', 'x', 2), ('v', 'y', 1), ('x', 'u', 3), ('x', 'v', 5), ('x', 'y', 2), ('y', 's', 7), ('y', 'v', 6)]
    XG = nx.Graph()
    XG.add_weighted_edges_from(edges)
    c = nx.closeness_centrality(XG, distance='weight')
    d = {'y': 0.2, 'x': 0.286, 's': 0.138, 'u': 0.235, 'v': 0.2}
    for n in sorted(XG):
        assert c[n] == pytest.approx(d[n], abs=0.001)