import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
def test_random_labeled_rooted_forest():
    for i in range(1, 10):
        t1 = nx.random_labeled_rooted_forest(i, seed=42)
        t2 = nx.random_labeled_rooted_forest(i, seed=42)
        assert nx.utils.misc.graphs_equal(t1, t2)
        for c in nx.connected_components(t1):
            assert nx.is_tree(t1.subgraph(c))
        assert 'root' not in t1.graph
        assert 'roots' in t1.graph