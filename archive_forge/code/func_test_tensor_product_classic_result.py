import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_tensor_product_classic_result():
    K2 = nx.complete_graph(2)
    G = nx.petersen_graph()
    G = nx.tensor_product(G, K2)
    assert nx.is_isomorphic(G, nx.desargues_graph())
    G = nx.cycle_graph(5)
    G = nx.tensor_product(G, K2)
    assert nx.is_isomorphic(G, nx.cycle_graph(10))
    G = nx.tetrahedral_graph()
    G = nx.tensor_product(G, K2)
    assert nx.is_isomorphic(G, nx.cubical_graph())