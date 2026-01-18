import os
import tempfile
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_read_pajek(self):
    G = nx.parse_pajek(self.data)
    Gin = nx.read_pajek(self.fname)
    assert sorted(G.nodes()) == sorted(Gin.nodes())
    assert edges_equal(G.edges(), Gin.edges())
    assert self.G.graph == Gin.graph
    for n in G:
        assert G.nodes[n] == Gin.nodes[n]