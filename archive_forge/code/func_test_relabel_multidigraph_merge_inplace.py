import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_multidigraph_merge_inplace(self):
    G = nx.MultiDiGraph([(0, 1), (0, 2), (0, 3)])
    G[0][1][0]['value'] = 'a'
    G[0][2][0]['value'] = 'b'
    G[0][3][0]['value'] = 'c'
    mapping = {1: 4, 2: 4, 3: 4}
    nx.relabel_nodes(G, mapping, copy=False)
    assert {'value': 'a'} in G[0][4].values()
    assert {'value': 'b'} in G[0][4].values()
    assert {'value': 'c'} in G[0][4].values()