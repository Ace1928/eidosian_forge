import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_single_edge_matching(self):
    G = nx.star_graph(5)
    matching = nx.maximal_matching(G)
    assert 1 == len(matching)
    assert nx.is_maximal_matching(G, matching)