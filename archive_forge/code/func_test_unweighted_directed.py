import networkx as nx
from networkx.algorithms.approximation import min_weighted_vertex_cover
def test_unweighted_directed(self):
    G = nx.DiGraph()
    G.add_edges_from(((0, v) for v in range(1, 26)))
    G.add_edges_from(((v, 0) for v in range(26, 51)))
    cover = min_weighted_vertex_cover(G)
    assert 1 == len(cover)
    assert is_cover(G, cover)