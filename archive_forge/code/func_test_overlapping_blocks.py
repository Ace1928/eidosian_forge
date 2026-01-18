import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_overlapping_blocks():
    with pytest.raises(nx.NetworkXException):
        G = nx.path_graph(6)
        partition = [{0, 1, 2}, {2, 3}, {4, 5}]
        nx.quotient_graph(G, partition)