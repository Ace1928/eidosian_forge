import pytest
import networkx as nx
from networkx.utils import pairwise
def test_unweighted_graph(self):
    G = nx.Graph()
    G.add_edges_from([(1, 0), (2, 1)])
    H = G.copy()
    nx.set_edge_attributes(H, values=1, name='weight')
    assert nx.johnson(G) == nx.johnson(H)