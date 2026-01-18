import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_self_loop(self):
    G = self.G()
    G.add_edge('A', 'A')
    assert G.has_edge('A', 'A')
    G.remove_edge('A', 'A')
    G.add_edge('X', 'X')
    assert G.has_node('X')
    G.remove_node('X')
    G.add_edge('A', 'Z')
    assert G.has_node('Z')