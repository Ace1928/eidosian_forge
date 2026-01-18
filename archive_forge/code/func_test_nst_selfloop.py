import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_selfloop(self):
    G = nx.complete_graph(3)
    G.add_edge(1, 1)
    assert np.isclose(nx.number_of_spanning_trees(G), 3)