import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_path_graph(self):
    G = nx.path_graph(5)
    assert np.isclose(nx.number_of_spanning_trees(G), 1)