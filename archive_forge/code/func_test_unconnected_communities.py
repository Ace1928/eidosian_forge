from itertools import chain, combinations
import pytest
import networkx as nx
def test_unconnected_communities():
    test = nx.Graph()
    test.add_edge('a', 'c')
    test.add_edge('a', 'd')
    test.add_edge('d', 'c')
    test.add_edge('b', 'e')
    test.add_edge('e', 'f')
    test.add_edge('f', 'b')
    ground_truth = {frozenset(['a', 'c', 'd']), frozenset(['b', 'e', 'f'])}
    communities = nx.community.label_propagation_communities(test)
    result = {frozenset(c) for c in communities}
    assert result == ground_truth