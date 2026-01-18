import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_order_size(self):
    G = self.G()
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'B'), ('C', 'D')])
    assert G.order() == 4
    assert G.size() == 5
    assert G.number_of_edges() == 5
    assert G.number_of_edges('A', 'B') == 1
    assert G.number_of_edges('A', 'D') == 0