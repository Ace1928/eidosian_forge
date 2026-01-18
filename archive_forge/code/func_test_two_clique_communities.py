import pytest
import networkx as nx
from networkx import Graph, NetworkXError
from networkx.algorithms.community import asyn_fluidc
def test_two_clique_communities():
    test = Graph()
    test.add_edge('a', 'b')
    test.add_edge('a', 'c')
    test.add_edge('b', 'c')
    test.add_edge('c', 'd')
    test.add_edge('d', 'e')
    test.add_edge('d', 'f')
    test.add_edge('f', 'e')
    ground_truth = {frozenset(['a', 'c', 'b']), frozenset(['e', 'd', 'f'])}
    communities = asyn_fluidc(test, 2, seed=7)
    result = {frozenset(c) for c in communities}
    assert result == ground_truth