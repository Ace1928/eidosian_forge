import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.utils import edges_equal, nodes_equal
def test_nonhashable_node(self):
    G = self.G()
    assert not G.has_node(['A'])
    assert not G.has_node({'A': 1})