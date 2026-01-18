import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_s_t_blossom(self):
    """Create S-blossom, relabel as T-blossom, use for augmentation:"""
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 9), (1, 3, 8), (2, 3, 10), (1, 4, 5), (4, 5, 4), (1, 6, 3)])
    answer = matching_dict_to_set({1: 6, 2: 3, 3: 2, 4: 5, 5: 4, 6: 1})
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)
    G.add_edge(4, 5, weight=3)
    G.add_edge(1, 6, weight=4)
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)
    G.remove_edge(1, 6)
    G.add_edge(3, 6, weight=4)
    answer = matching_dict_to_set({1: 2, 2: 1, 3: 6, 4: 5, 5: 4, 6: 3})
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)