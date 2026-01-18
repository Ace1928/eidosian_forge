import pytest
import networkx as nx
def test_simple_directed_configuration_model():
    G = nx.directed_configuration_model([1, 1], [1, 1], seed=0)
    assert len(G) == 2