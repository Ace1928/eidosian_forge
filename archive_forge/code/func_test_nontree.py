from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nontree(self):
    with pytest.raises(nx.NotATree):
        G = nx.cycle_graph(3)
        nx.to_nested_tuple(G, 0)