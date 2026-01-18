from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_unknown_root(self):
    with pytest.raises(nx.NodeNotFound):
        G = nx.path_graph(2)
        nx.to_nested_tuple(G, 'bogus')