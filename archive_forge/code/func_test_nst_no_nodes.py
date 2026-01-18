import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_nst_no_nodes(self):
    G = nx.Graph()
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.number_of_spanning_trees(G)