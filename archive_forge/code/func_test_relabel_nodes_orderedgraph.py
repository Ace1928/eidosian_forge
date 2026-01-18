import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_relabel_nodes_orderedgraph(self):
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([(1, 3), (2, 3)])
    mapping = {1: 'a', 2: 'b', 3: 'c'}
    H = nx.relabel_nodes(G, mapping)
    assert list(H.nodes) == ['a', 'b', 'c']