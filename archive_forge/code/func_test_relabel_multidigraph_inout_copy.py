import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_multidigraph_inout_copy(self):
    G = nx.MultiDiGraph([(0, 4), (1, 4), (4, 2), (4, 3)])
    G[0][4][0]['value'] = 'a'
    G[1][4][0]['value'] = 'b'
    G[4][2][0]['value'] = 'c'
    G[4][3][0]['value'] = 'd'
    G.add_edge(0, 4, key='x', value='e')
    G.add_edge(4, 3, key='x', value='f')
    mapping = {0: 9, 1: 9, 2: 9, 3: 9}
    H = nx.relabel_nodes(G, mapping, copy=True)
    assert {'value': 'a'} in H[9][4].values()
    assert {'value': 'b'} in H[9][4].values()
    assert {'value': 'c'} in H[4][9].values()
    assert len(H[4][9]) == 3
    assert {'value': 'd'} in H[4][9].values()
    assert {'value': 'e'} in H[9][4].values()
    assert {'value': 'f'} in H[4][9].values()
    assert len(H[9][4]) == 3