import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_str_unnamed(self):
    G = self.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    assert str(G) == f'{type(G).__name__} with 3 nodes and 2 edges'