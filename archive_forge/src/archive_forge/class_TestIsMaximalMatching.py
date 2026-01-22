import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
class TestIsMaximalMatching:
    """Unit tests for the
    :func:`~networkx.algorithms.matching.is_maximal_matching` function.

    """

    def test_dict(self):
        G = nx.path_graph(4)
        assert nx.is_maximal_matching(G, {0: 1, 1: 0, 2: 3, 3: 2})

    def test_invalid_input(self):
        error = nx.NetworkXError
        G = nx.path_graph(4)
        raises(error, nx.is_maximal_matching, G, {(0, 5)})
        raises(error, nx.is_maximal_matching, G, {(5, 0)})
        raises(error, nx.is_maximal_matching, G, {(0, 1, 2), (2, 3)})
        raises(error, nx.is_maximal_matching, G, {(0,), (2, 3)})

    def test_valid(self):
        G = nx.path_graph(4)
        assert nx.is_maximal_matching(G, {(0, 1), (2, 3)})

    def test_not_matching(self):
        G = nx.path_graph(4)
        assert not nx.is_maximal_matching(G, {(0, 1), (1, 2), (2, 3)})
        assert not nx.is_maximal_matching(G, {(0, 3)})
        G.add_edge(0, 0)
        assert not nx.is_maximal_matching(G, {(0, 0)})

    def test_not_maximal(self):
        G = nx.path_graph(4)
        assert not nx.is_maximal_matching(G, {(0, 1)})