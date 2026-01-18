import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_s_blossom(self):
    """Create S-blossom and use it for augmentation:"""
    G = nx.Graph()
    G.add_weighted_edges_from([(1, 2, 8), (1, 3, 9), (2, 3, 10), (3, 4, 7)])
    answer = matching_dict_to_set({1: 2, 2: 1, 3: 4, 4: 3})
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)
    G.add_weighted_edges_from([(1, 6, 5), (4, 5, 6)])
    answer = matching_dict_to_set({1: 6, 2: 3, 3: 2, 4: 5, 5: 4, 6: 1})
    assert edges_equal(nx.max_weight_matching(G), answer)
    assert edges_equal(nx.min_weight_matching(G), answer)