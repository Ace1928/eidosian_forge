import networkx as nx
from networkx.algorithms.approximation import min_weighted_vertex_cover
def test_unweighted_undirected(self):
    size = 50
    sg = nx.star_graph(size)
    cover = min_weighted_vertex_cover(sg)
    assert 1 == len(cover)
    assert is_cover(sg, cover)