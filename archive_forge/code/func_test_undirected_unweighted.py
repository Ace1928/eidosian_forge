import networkx as nx
from networkx.utils import pairwise
def test_undirected_unweighted(self):
    G = nx.cycle_graph(6)
    cells = nx.voronoi_cells(G, {0, 3})
    expected = {0: {0, 1, 5}, 3: {2, 3, 4}}
    assert expected == cells