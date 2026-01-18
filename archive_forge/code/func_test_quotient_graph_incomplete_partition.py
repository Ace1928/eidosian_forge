import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_quotient_graph_incomplete_partition():
    G = nx.path_graph(6)
    partition = []
    H = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(H.nodes(), [])
    assert edges_equal(H.edges(), [])
    partition = [[0, 1], [2, 3], [5]]
    H = nx.quotient_graph(G, partition, relabel=True)
    assert nodes_equal(H.nodes(), [0, 1, 2])
    assert edges_equal(H.edges(), [(0, 1)])