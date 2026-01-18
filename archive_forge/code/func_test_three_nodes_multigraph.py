from math import sqrt
import pytest
import networkx as nx
@pytest.mark.parametrize('method', methods)
def test_three_nodes_multigraph(self, method):
    pytest.importorskip('scipy')
    G = nx.MultiDiGraph()
    G.add_weighted_edges_from([(1, 2, 1), (1, 3, 2), (2, 3, 1), (2, 3, 2)])
    order = nx.spectral_ordering(G, method=method)
    assert set(order) == set(G)
    assert {2, 3} in (set(order[:-1]), set(order[1:]))