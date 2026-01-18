import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_multidigraph_merge_copy(self):
    G = nx.MultiDiGraph([(0, 1), (0, 2), (0, 3)])
    G[0][1][0]['value'] = 'a'
    G[0][2][0]['value'] = 'b'
    G[0][3][0]['value'] = 'c'
    mapping = {1: 4, 2: 4, 3: 4}
    H = nx.relabel_nodes(G, mapping, copy=True)
    assert {'value': 'a'} in H[0][4].values()
    assert {'value': 'b'} in H[0][4].values()
    assert {'value': 'c'} in H[0][4].values()