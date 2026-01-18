import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_trivial1(self):
    """Empty graph"""
    G = nx.Graph()
    assert nx.max_weight_matching(G) == set()
    assert nx.min_weight_matching(G) == set()