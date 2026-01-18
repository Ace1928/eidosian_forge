import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_strong_product_combinations():
    P5 = nx.path_graph(5)
    K3 = nx.complete_graph(3)
    G = nx.strong_product(P5, K3)
    assert nx.number_of_nodes(G) == 5 * 3
    G = nx.strong_product(nx.MultiGraph(P5), K3)
    assert nx.number_of_nodes(G) == 5 * 3
    G = nx.strong_product(P5, nx.MultiGraph(K3))
    assert nx.number_of_nodes(G) == 5 * 3
    G = nx.strong_product(nx.MultiGraph(P5), nx.MultiGraph(K3))
    assert nx.number_of_nodes(G) == 5 * 3