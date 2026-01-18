import pytest
import networkx as nx
from networkx.algorithms.approximation.steinertree import metric_closure, steiner_tree
from networkx.utils import edges_equal
def test_steiner_tree_method_invalid():
    G = nx.star_graph(4)
    with pytest.raises(ValueError, match='invalid_method is not a valid choice for an algorithm.'):
        nx.approximation.steiner_tree(G, terminal_nodes=[1, 3], method='invalid_method')