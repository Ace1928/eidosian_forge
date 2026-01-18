import os
import tempfile
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_parse_pajek_simple(self):
    data = '*Vertices 2\n1 "1"\n2 "2"\n*Edges\n1 2\n2 1'
    G = nx.parse_pajek(data)
    assert sorted(G.nodes()) == ['1', '2']
    assert edges_equal(G.edges(), [('1', '2'), ('1', '2')])