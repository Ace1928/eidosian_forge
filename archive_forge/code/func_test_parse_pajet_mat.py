import os
import tempfile
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_parse_pajet_mat(self):
    data = '*Vertices 3\n1 "one"\n2 "two"\n3 "three"\n*Matrix\n1 1 0\n0 1 0\n0 1 0\n'
    G = nx.parse_pajek(data)
    assert set(G.nodes()) == {'one', 'two', 'three'}
    assert G.nodes['two'] == {'id': '2'}
    assert edges_equal(set(G.edges()), {('one', 'one'), ('two', 'one'), ('two', 'two'), ('two', 'three')})