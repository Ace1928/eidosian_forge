import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
def test_random_unlabeled_rooted_tree():
    for i in range(1, 10):
        t1 = nx.random_unlabeled_rooted_tree(i, seed=42)
        t2 = nx.random_unlabeled_rooted_tree(i, seed=42)
        assert nx.utils.misc.graphs_equal(t1, t2)
        assert nx.is_tree(t1)
        assert 'root' in t1.graph
        assert 'roots' not in t1.graph
    t = nx.random_unlabeled_rooted_tree(15, number_of_trees=10, seed=43)
    random.seed(43)
    s = nx.random_unlabeled_rooted_tree(15, number_of_trees=10, seed=random)
    for i in range(10):
        assert nx.utils.misc.graphs_equal(t[i], s[i])
        assert nx.is_tree(t[i])
        assert 'root' in t[i].graph
        assert 'roots' not in t[i].graph