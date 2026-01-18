import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
def test_random_unlabeled_rooted_forest():
    with pytest.raises(ValueError):
        nx.random_unlabeled_rooted_forest(10, q=0, seed=42)
    for i in range(1, 10):
        for q in range(1, i + 1):
            t1 = nx.random_unlabeled_rooted_forest(i, q=q, seed=42)
            t2 = nx.random_unlabeled_rooted_forest(i, q=q, seed=42)
            assert nx.utils.misc.graphs_equal(t1, t2)
            for c in nx.connected_components(t1):
                assert nx.is_tree(t1.subgraph(c))
                assert len(c) <= q
            assert 'root' not in t1.graph
            assert 'roots' in t1.graph
    t = nx.random_unlabeled_rooted_forest(15, number_of_forests=10, seed=43)
    random.seed(43)
    s = nx.random_unlabeled_rooted_forest(15, number_of_forests=10, seed=random)
    for i in range(10):
        assert nx.utils.misc.graphs_equal(t[i], s[i])
        for c in nx.connected_components(t[i]):
            assert nx.is_tree(t[i].subgraph(c))
        assert 'root' not in t[i].graph
        assert 'roots' in t[i].graph